"""Approved catalogue lookup and indicative quotes, without model arithmetic.

The authoring Markdown and generated PDF are deliberately structured. Unexpected/missing rows fail
startup rather than silently supplying a partial catalogue or fallback prices.
Changing the supported catalogue dimensions requires reviewing this parser too.
"""

from dataclasses import asdict, dataclass
from decimal import Decimal
from itertools import product
from pathlib import Path
import re
from types import MappingProxyType


class CatalogueError(ValueError):
    """The approved source is missing, malformed, or internally inconsistent."""


CHOICES = {
    "design": ("axis", "orbit", "offset"),
    "size_tier": ("compact", "standard", "large"),
    "format": ("square", "portrait", "landscape"),
    "surface": ("smooth_square", "studded_square", "round"),
    "product_form": ("diy", "completed"),
}
_FORMATS = {"Square (1:1)": "square", "Portrait (3:4)": "portrait",
            "Landscape (4:3)": "landscape"}
_SURFACES = {"Smooth square": "smooth_square", "Studded square": "studded_square",
             "Round": "round"}
_GRID_HEADER = ("Size tier", "Format", "Grid in blocks", "Visible pieces",
                "Artwork dimensions in cm")
_PRICE_HEADER = ("Size tier", "Format", "Grid", "Block surface", "DIY kit (USD)",
                 "Completed piece (USD)")


def _section(text: str, heading: str) -> str:
    matches = list(re.finditer(r"^## " + re.escape(heading) + r"\s*$", text, re.MULTILINE))
    if len(matches) != 1:
        raise CatalogueError(f"Expected one {heading} section in the master.")
    rest = text[matches[0].end():]
    return re.split(r"^## ", rest, maxsplit=1, flags=re.MULTILINE)[0]


def _table(section: str, expected_header: tuple[str, ...]) -> list[tuple[str, ...]]:
    lines = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]
    cells = [tuple(cell.strip() for cell in line.strip("|").split("|")) for line in lines]
    if (len(cells) < 3 or cells[0] != expected_header
            or len(cells[1]) != len(expected_header)
            or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells[1])
            or any(len(row) != len(expected_header) for row in cells[2:])):
        raise CatalogueError("The master has an unexpected catalogue table structure.")
    return cells[2:]


def _positive_int(text: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", text) is None:
        raise CatalogueError("Catalogue numbers must be positive whole numbers.")
    return int(text)


def _grid(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]*) × ([1-9][0-9]*)", text)
    if match is None:
        raise CatalogueError("A catalogue grid must specify width × height.")
    return tuple(map(int, match.groups()))


def _tier_format(tier: str, format_label: str) -> tuple[str, str]:
    tier = tier.lower()
    if tier not in CHOICES["size_tier"] or format_label not in _FORMATS:
        raise CatalogueError("The master uses an unsupported tier or format.")
    return tier, _FORMATS[format_label]


@dataclass(frozen=True)
class ProductOption:
    product_id: str
    design: str
    size_tier: str
    format: str
    surface: str
    product_form: str
    grid_width: int
    grid_height: int
    visible_pieces: int
    artwork_width_cm: float
    artwork_height_cm: float
    unit_price_usd: int


def pdf_catalogue_source(text: str) -> str:
    """Translate the known business PDF layout into validator input, in memory.

    Missing/changed table rows fail the existing completeness and price checks.
    No model interpretation or fallback prices participate in this conversion.
    """
    text = re.sub(r"(?<=\d) x (?=\d)", " × ", text)
    text = re.sub(r"\s+", " ", text)
    def section(start, end):
        if text.count(start) != 1 or text.count(end) != 1:
            raise CatalogueError("The business PDF has missing or repeated sections.")
        return text.split(start, 1)[1].split(end, 1)[0]
    products = section("Products and services", "Predefined designs")
    designs = section("Predefined designs", "Approved prices")
    prices = section("Approved prices", "Approved payment and delivery terms")
    terms = section("Approved payment and delivery terms", "Approved return and damaged-delivery terms")
    tier = r"(Compact|Standard|Large)"
    format_label = r"(Square \(1:1\)|Portrait \(3:4\)|Landscape \(4:3\))"
    grid = r"([0-9]+ × [0-9]+)"
    grid_rows = re.findall(tier + r" " + format_label + r" " + grid + r" ([0-9,]+) ([0-9]+\.[0-9]+ × [0-9]+\.[0-9]+)", products)
    price_rows = re.findall(tier + r" " + format_label + r" " + grid + r" (Smooth square|Studded square|Round) ([0-9]+) ([0-9]+)", prices)
    descriptions = re.findall(r"- (Axis|Orbit|Offset): (.*?)(?= - (?:Axis|Orbit|Offset):| Each design)", designs)
    def table(header, rows):
        return "\n".join("| " + " | ".join(row) + " |" for row in (header, ("---",) * len(header), *rows))
    delivery = re.search(r"- Delivery fee: USD ([0-9]+) per order\.", terms)
    if delivery is None:
        raise CatalogueError("The business PDF has no delivery fee.")
    return ("## Products and services\n" + products + "\n" + table(_GRID_HEADER, grid_rows)
        + "\n## Predefined designs\n" + "\n".join(f"- **{name}:** {description}" for name, description in descriptions)
        + "\n## Approved prices\n" + prices + "\n" + table(_PRICE_HEADER, price_rows)
        + "\n## Approved payment and delivery terms\n- Delivery fee: USD " + delivery.group(1) + " per order.\n")


class Catalogue:
    def __init__(self, master_path: str | Path):
        try:
            text = Path(master_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise CatalogueError("The approved business master could not be read.") from exc
        self._load(text)

    @classmethod
    def from_pdf_text(cls, text: str):
        """Load the approved PDF's extracted text through the same validators."""
        catalogue = cls.__new__(cls)
        catalogue._load(pdf_catalogue_source(text))
        return catalogue

    def _load(self, text: str) -> None:
        section = _section(text, "Products and services")
        spacing = re.search(r"Block spacing is ([1-9][0-9]*) mm", section)
        if spacing is None:
            raise CatalogueError("The master must define block spacing.")
        pitch_cm = Decimal(spacing.group(1)) / 10
        grids = {}
        for tier, format_label, grid_text, pieces_text, dimensions in _table(section, _GRID_HEADER):
            key = _tier_format(tier, format_label)
            width, height = _grid(grid_text)
            pieces = _positive_int(pieces_text.replace(",", ""))
            cm = re.fullmatch(r"([0-9]+\.[0-9]+) × ([0-9]+\.[0-9]+)", dimensions)
            if (key in grids or pieces != width * height or cm is None
                    or tuple(map(Decimal, cm.groups())) != (width * pitch_cm, height * pitch_cm)):
                raise CatalogueError("Duplicate or inconsistent grid measurements in the master.")
            ratio = {"square": (1, 1), "portrait": (3, 4), "landscape": (4, 3)}[key[1]]
            if width * ratio[1] != height * ratio[0]:
                raise CatalogueError("The grid does not match its stated aspect ratio.")
            grids[key] = (width, height, pieces, float(width * pitch_cm), float(height * pitch_cm))
        expected_grids = set(product(CHOICES["size_tier"], CHOICES["format"]))
        if set(grids) != expected_grids:
            raise CatalogueError("The master must contain all nine grid configurations.")

        descriptions = {}
        for line in _section(text, "Predefined designs").splitlines():
            if not line.startswith("- "):
                continue
            match = re.fullmatch(r"- \*\*(Axis|Orbit|Offset):\*\* (.+)", line)
            if match is None or match.group(1).lower() in descriptions:
                raise CatalogueError("Unexpected or duplicate design description in the master.")
            descriptions[match.group(1).lower()] = match.group(2)
        if set(descriptions) != set(CHOICES["design"]):
            raise CatalogueError("The master must describe all three approved designs.")

        prices = {}
        price_section = _section(text, "Approved prices")
        for tier, format_label, grid_text, surface_label, diy, completed in _table(price_section, _PRICE_HEADER):
            tier_key, format_key = _tier_format(tier, format_label)
            if surface_label not in _SURFACES:
                raise CatalogueError("The master uses an unsupported block surface.")
            key = (tier_key, format_key, _SURFACES[surface_label])
            amounts = (_positive_int(diy), _positive_int(completed))
            if key in prices or _grid(grid_text) != grids[(tier_key, format_key)][:2]:
                raise CatalogueError("Duplicate price row or conflicting grid in the master.")
            if amounts[1] <= amounts[0]:
                raise CatalogueError("Completed-piece prices must include an assembly fee.")
            prices[key] = amounts
        if set(prices) != set(product(CHOICES["size_tier"], CHOICES["format"], CHOICES["surface"])):
            raise CatalogueError("The master must contain all 27 approved price rows.")
        fees = re.findall(r"USD ([1-9][0-9]*) (Compact|Standard|Large)", price_section)
        assembly = {tier.lower(): int(amount) for amount, tier in fees}
        if len(fees) != 3 or set(assembly) != set(CHOICES["size_tier"]):
            raise CatalogueError("The master must define each tier's assembly fee.")
        for (tier, format_key, surface), amounts in prices.items():
            if amounts[1] - amounts[0] != assembly[tier]:
                raise CatalogueError("A completed price conflicts with its assembly fee.")
            if format_key == "portrait" and amounts != prices[(tier, "landscape", surface)]:
                raise CatalogueError("Portrait and landscape prices must agree.")
        delivery = re.search(r"only additional checkout charge is USD ([1-9][0-9]*) delivery per order", price_section)
        delivery_terms = re.search(r"^- Delivery fee: USD ([1-9][0-9]*) per order\.$",
                                   _section(text, "Approved payment and delivery terms"), re.MULTILINE)
        if delivery is None or delivery_terms is None or delivery.group(1) != delivery_terms.group(1):
            raise CatalogueError("The master must define a consistent per-order delivery fee.")

        options = []
        for design, tier, format_key, surface, form in product(*CHOICES.values()):
            width, height, pieces, width_cm, height_cm = grids[(tier, format_key)]
            price = prices[(tier, format_key, surface)][0 if form == "diy" else 1]
            product_id = ":".join((design, tier, format_key, surface, form))
            options.append(ProductOption(product_id, design, tier, format_key, surface, form,
                                         width, height, pieces, width_cm, height_cm, price))
        self.options = tuple(options)
        self.by_id = MappingProxyType({option.product_id: option for option in options})
        self.design_descriptions = MappingProxyType(descriptions)
        self.delivery_fee_usd = int(delivery.group(1))

    def get_product_options(self, *, design=None, size_tier=None, format=None,
                            surface=None, product_form=None) -> dict:
        filters = {"design": design, "size_tier": size_tier, "format": format,
                   "surface": surface, "product_form": product_form}
        for key, value in filters.items():
            if value is not None and (not isinstance(value, str) or value not in CHOICES[key]):
                return error("invalid_arguments", f"Use an approved {key} value or null.")
        matches = [asdict(option) for option in self.options
                   if all(value is None or getattr(option, key) == value for key, value in filters.items())]
        return {"ok": True, "currency": "USD", "count": len(matches), "options": matches,
                "design_descriptions": dict(self.design_descriptions),
                "notice": "Catalogue options do not establish live stock. Dimensions exclude frames."}

    def calculate_quote(self, items: object) -> dict:
        if not isinstance(items, list) or not items:
            return error("invalid_arguments", "Supply at least one product ID and positive whole quantity.")
        quantities = {}
        for item in items:
            if not isinstance(item, dict) or set(item) != {"product_id", "quantity"}:
                return error("invalid_arguments", "Each item accepts only product_id and quantity.")
            product_id, quantity = item["product_id"], item["quantity"]
            if not isinstance(product_id, str) or product_id not in self.by_id:
                return error("unknown_product", "Use a product_id returned by get_product_options.")
            if type(quantity) is not int or quantity < 1:
                return error("invalid_arguments", "Quantity must be a positive whole number.")
            quantities[product_id] = quantities.get(product_id, 0) + quantity
        lines = []
        for product_id, quantity in quantities.items():
            option = self.by_id[product_id]
            lines.append({"product": asdict(option), "quantity": quantity,
                          "line_total_usd": option.unit_price_usd * quantity})
        subtotal = sum(line["line_total_usd"] for line in lines)
        return {"ok": True, "status": "indicative_quote", "currency": "USD", "items": lines,
                "subtotal_usd": subtotal, "delivery_usd": self.delivery_fee_usd,
                "total_usd": subtotal + self.delivery_fee_usd,
                "notice": "This estimate does not place an order or establish live stock."}


def error(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


GET_PRODUCT_OPTIONS_SCHEMA = {
    "type": "function", "function": {
        "name": "get_product_options",
        "description": "Look up approved catalogue options. Use null for unspecified preferences; use returned IDs for quotes. This is not a stock check.",
        "strict": True,
        "parameters": {"type": "object", "properties": {
            key: {"type": ["string", "null"], "enum": [*values, None]}
            for key, values in CHOICES.items()
        }, "required": list(CHOICES), "additionalProperties": False},
    },
}

CALCULATE_QUOTE_SCHEMA = {
    "type": "function", "function": {
        "name": "calculate_quote",
        "description": "Calculate an indicative USD quote from approved product IDs and quantities. Applies delivery once; does not place an order.",
        "strict": True,
        "parameters": {"type": "object", "properties": {
            "items": {"type": "array", "minItems": 1, "items": {
                "type": "object", "properties": {
                    "product_id": {"type": "string", "description": "Exact product_id from get_product_options."},
                    "quantity": {"type": "integer", "minimum": 1},
                }, "required": ["product_id", "quantity"], "additionalProperties": False,
            }},
        }, "required": ["items"], "additionalProperties": False},
    },
}
