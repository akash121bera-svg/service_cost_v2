"""
Semantic cost decomposition helpers.

This module maps business cost language such as handling, audit, or insurance
to the deterministic CSV columns that already feed the costing engine.
"""

from difflib import get_close_matches
import re

from engine.category_selector import get_shipment_category
from engine.costing_engine import calculate_total_cost
from engine.uploaded_costs import (
    SERVICE_RATE_NAMES,
    find_vendor_matches,
    get_factor_rate_value,
    get_vendor_name,
)


COST_COMPONENT_SCHEMA = {
    "packaging": {
        "label": "Packaging",
        "aggregate_rate": "packaging_rate",
        "components": {
            "pouch_rate": {
                "label": "Pouch Rate",
                "column": "pouch_rate_per_unit",
                "divisor": "unit",
                "aliases": ["pouch", "pouch rate"],
            },
            "label_rate": {
                "label": "Label Rate",
                "column": "label_rate_per_unit",
                "divisor": "unit",
                "aliases": ["label", "label rate"],
            },
            "carton_rate": {
                "label": "Carton Rate",
                "column": "carton_rate_per_unit",
                "divisor": "unit",
                "aliases": ["carton", "carton rate"],
            },
        },
    },
    "sterilization": {
        "label": "Sterilization",
        "aggregate_rate": "sterilization_rate",
        "components": {
            "sterilization_batch_cost": {
                "label": "Sterilization Batch Cost",
                "column": "sterilization_batch_cost",
                "divisor": "units_per_batch",
                "aliases": ["sterilization batch", "batch sterilization"],
            },
            "validation_cost": {
                "label": "Validation Cost",
                "column": "validation_cost",
                "divisor": "units_per_batch",
                "aliases": ["validation", "validation cost"],
            },
        },
    },
    "logistics": {
        "label": "Logistics",
        "aggregate_rate": "logistics_rate",
        "components": {
            "transport_cost": {
                "label": "Transport Cost",
                "column": "transport_cost_per_shipment",
                "divisor": "quantity",
                "aliases": ["transport", "transportation", "freight", "shipping"],
            },
            "handling_cost": {
                "label": "Handling Cost",
                "column": "handling_cost_per_shipment",
                "divisor": "quantity",
                "aliases": ["handling", "handling cost"],
            },
            "distributor_fee": {
                "label": "Distributor Fee",
                "column": "distributor_fee_per_shipment",
                "divisor": "quantity",
                "aliases": ["distributor", "distributor fee", "distribution"],
            },
        },
    },
    "quality": {
        "label": "Quality",
        "aggregate_rate": "quality_rate",
        "components": {
            "inspection_cost": {
                "label": "Inspection Cost",
                "column": "inspection_cost_per_shipment",
                "divisor": "quantity",
                "aliases": ["inspection", "inspection cost"],
            },
            "audit_cost": {
                "label": "Audit Cost",
                "column": "audit_cost_per_shipment",
                "divisor": "quantity",
                "aliases": ["audit", "audit cost", "quality audit"],
            },
            "documentation_cost": {
                "label": "Documentation Cost",
                "column": "documentation_cost_per_shipment",
                "divisor": "quantity",
                "aliases": ["documentation", "document", "documents"],
            },
        },
    },
    "warehousing": {
        "label": "Warehousing",
        "aggregate_rate": "warehousing_rate",
        "components": {
            "warehouse_rent": {
                "label": "Warehouse Rent",
                "column": "warehouse_rent_period",
                "divisor": "quantity",
                "aliases": ["warehouse", "warehouse rent", "rent"],
            },
            "insurance_cost": {
                "label": "Insurance Cost",
                "column": "insurance_cost_period",
                "divisor": "quantity",
                "aliases": ["insurance", "insurance cost"],
            },
            "inventory_handling": {
                "label": "Inventory Handling",
                "column": "inventory_handling_cost_period",
                "divisor": "quantity",
                "aliases": ["inventory", "inventory handling", "storage handling"],
            },
        },
    },
}


SEMANTIC_COST_MAP = {}

for category_name, category_schema in COST_COMPONENT_SCHEMA.items():
    SEMANTIC_COST_MAP[category_name] = {
        "category": category_name,
        "component": None,
    }

    for component_name, component_schema in category_schema["components"].items():
        SEMANTIC_COST_MAP[component_name] = {
            "category": category_name,
            "component": component_name,
        }

        for alias in component_schema["aliases"]:
            SEMANTIC_COST_MAP[alias] = {
                "category": category_name,
                "component": component_name,
            }


DETAIL_TRIGGER_TERMS = [
    "breakdown",
    "detailed",
    "detail",
    "decompose",
    "decomposition",
    "component",
    "components",
    "sub-component",
    "subcomponent",
]


def is_detailed_audit_question(question_lower):
    """Return True when a question needs hierarchical decomposition."""
    if any(term in question_lower for term in DETAIL_TRIGGER_TERMS):
        return True

    return any(
        mapping["component"] is not None
        for mapping in get_requested_semantic_costs(question_lower).values()
    )


def get_requested_semantic_costs(question_lower):
    """Map business-language cost terms in the question to schema entries."""
    matches = {}
    normalized_question = re.sub(r"[^a-z0-9\s]", " ", question_lower)
    question_tokens = set(normalized_question.split())

    for alias, mapping in SEMANTIC_COST_MAP.items():
        normalized_alias = re.sub(r"[^a-z0-9\s]", " ", alias).strip()

        if " " in normalized_alias:
            found = normalized_alias in normalized_question
        else:
            found = normalized_alias in question_tokens

        if found:
            matches[alias] = mapping

    if matches:
        return matches

    for token in question_tokens:
        close_matches = get_close_matches(
            token,
            SEMANTIC_COST_MAP.keys(),
            n=1,
            cutoff=0.86,
        )

        if close_matches:
            alias = close_matches[0]
            matches[alias] = SEMANTIC_COST_MAP[alias]

    return matches


def get_requested_categories(question_lower):
    """Return the parent categories that should be expanded."""
    semantic_matches = get_requested_semantic_costs(question_lower)
    has_detail_trigger = any(term in question_lower for term in DETAIL_TRIGGER_TERMS)
    categories = {
        mapping["category"]
        for mapping in semantic_matches.values()
        if has_detail_trigger or mapping["component"] is not None
    }

    if has_detail_trigger:
        for rate_name in SERVICE_RATE_NAMES:
            if rate_name in question_lower:
                categories.add(rate_name)

        if not categories:
            return list(COST_COMPONENT_SCHEMA.keys())

    return [
        category
        for category in COST_COMPONENT_SCHEMA
        if category in categories
    ]


def _component_value(row, component_schema, quantity):
    column_name = component_schema["column"]

    if column_name not in row:
        return None

    raw_value = float(row[column_name])
    divisor = component_schema["divisor"]

    if divisor == "unit":
        return raw_value

    if divisor == "quantity":
        return raw_value / quantity

    if divisor in row and float(row[divisor]) != 0:
        return raw_value / float(row[divisor])

    return None


def build_semantic_breakdown_rows(question, csv_dataframes, quantity):
    """Build hierarchical rows for detailed procurement audit questions."""
    question_lower = question.lower()
    vendor_matches = find_vendor_matches(question, csv_dataframes)
    selected_dataframes = vendor_matches or csv_dataframes
    requested_categories = get_requested_categories(question_lower)

    if not requested_categories:
        return "", []

    table_rows = []

    for file_name, df in selected_dataframes:
        row = get_shipment_category(df, quantity)
        total_cost = calculate_total_cost(row, quantity)

        for category_name in requested_categories:
            category_schema = COST_COMPONENT_SCHEMA[category_name]
            aggregate_rate = get_factor_rate_value(row, category_name, quantity)

            category_row_count = 0

            for component_name, component_schema in category_schema["components"].items():
                per_unit_value = _component_value(row, component_schema, quantity)
                availability = "available" if per_unit_value is not None else "missing"

                table_rows.append(
                    {
                        "response_mode": "detailed_audit",
                        "vendor": get_vendor_name(file_name),
                        "file": file_name,
                        "shipment_category": row.get("shipment_category", "N/A"),
                        "quantity": quantity,
                        "category": category_schema["label"],
                        "component": component_schema["label"],
                        "source_column": component_schema["column"],
                        "availability": availability,
                        "per_unit_cost": round(per_unit_value, 2) if per_unit_value is not None else None,
                        "aggregate_rate": round(aggregate_rate, 2) if aggregate_rate is not None else None,
                        "total_cost": float(total_cost),
                    }
                )
                category_row_count += 1

            if category_row_count == 0:
                table_rows.append(
                    {
                        "response_mode": "detailed_audit",
                        "vendor": get_vendor_name(file_name),
                        "file": file_name,
                        "shipment_category": row.get("shipment_category", "N/A"),
                        "quantity": quantity,
                        "category": category_schema["label"],
                        "component": "No component columns available",
                        "source_column": "",
                        "availability": "missing",
                        "per_unit_cost": None,
                        "aggregate_rate": round(aggregate_rate, 2) if aggregate_rate is not None else None,
                        "total_cost": float(total_cost),
                    }
                )

    return build_semantic_explanation(question_lower, table_rows), table_rows


def build_semantic_breakdown_rows_by_category(question, csv_dataframes):
    """Build detailed rows for every shipment category when no quantity is named."""
    question_lower = question.lower()
    vendor_matches = find_vendor_matches(question, csv_dataframes)
    selected_dataframes = vendor_matches or csv_dataframes
    requested_categories = get_requested_categories(question_lower)

    if not requested_categories:
        return "", []

    table_rows = []

    for file_name, df in selected_dataframes:
        for _, row in df.iterrows():
            quantity_basis = int(row.get("min_qty", 1))
            total_cost = calculate_total_cost(row, quantity_basis)

            for category_name in requested_categories:
                category_schema = COST_COMPONENT_SCHEMA[category_name]
                aggregate_rate = get_factor_rate_value(row, category_name, quantity_basis)

                for _, component_schema in category_schema["components"].items():
                    per_unit_value = _component_value(row, component_schema, quantity_basis)
                    availability = "available" if per_unit_value is not None else "missing"

                    table_rows.append(
                        {
                            "response_mode": "detailed_audit",
                            "vendor": get_vendor_name(file_name),
                            "file": file_name,
                            "shipment_category": row.get("shipment_category", "N/A"),
                            "category_min_qty": int(row.get("min_qty", quantity_basis)),
                            "category_max_qty": int(row.get("max_qty", quantity_basis)),
                            "quantity": quantity_basis,
                            "quantity_basis": f"category min_qty {quantity_basis}",
                            "category": category_schema["label"],
                            "component": component_schema["label"],
                            "source_column": component_schema["column"],
                            "availability": availability,
                            "per_unit_cost": round(per_unit_value, 2) if per_unit_value is not None else None,
                            "aggregate_rate": round(aggregate_rate, 2) if aggregate_rate is not None else None,
                            "total_cost": float(total_cost),
                        }
                    )

    answer = build_semantic_explanation(question_lower, table_rows)
    answer += (
        " No single product quantity was specified, so the table is grouped by "
        "shipment category and uses each category's min_qty as the denominator "
        "for per-unit shipment and period costs."
    )

    return answer, table_rows


def build_semantic_explanation(question_lower, table_rows):
    """Create an executive explanation for semantic-to-aggregate mappings."""
    if not table_rows:
        return ""

    available_categories = []
    partial_categories = []
    missing_terms = []

    for category in sorted({row["category"] for row in table_rows}):
        category_rows = [row for row in table_rows if row["category"] == category]
        available_count = sum(row["availability"] == "available" for row in category_rows)

        if available_count == len(category_rows):
            available_categories.append(category)
        elif available_count:
            partial_categories.append(category)
        else:
            missing_terms.append(category)

    relationship_lines = []
    category_lookup = {
        schema["label"]: category_name
        for category_name, schema in COST_COMPONENT_SCHEMA.items()
    }

    for category_label in available_categories + partial_categories:
        category_name = category_lookup[category_label]
        component_labels = [
            component_schema["label"]
            for component_schema in COST_COMPONENT_SCHEMA[category_name]["components"].values()
        ]
        relationship_lines.append(
            f"{category_label} expands into " + ", ".join(component_labels) + "."
        )

    answer = (
        "Detailed procurement audit view. Semantic cost terms are mapped to the "
        "deterministic cost hierarchy while preserving the aggregated service buckets. "
        + " ".join(relationship_lines)
    )

    if partial_categories:
        answer += (
            " Some requested categories are partially available; missing fields are "
            "marked in the table."
        )

    if missing_terms:
        answer += (
            " The requested category data is not present for: "
            + ", ".join(missing_terms)
            + "."
        )

    return answer
