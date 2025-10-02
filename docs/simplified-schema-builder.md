# Simplified Schema Builder

## Overview

The simplified schema builder uses **JSON Schema** as the primary format with
IDP-specific extensions stored in `x-idp-*` fields.

## Key Improvements

### 1. **JSON Schema as Primary Format**

- Industry standard format
- Better tooling support
- Easier to validate and share

### 2. **IDP-Specific Extensions**

All IDP configuration is stored in JSON Schema extension fields:

```json
{
  "type": "object",
  "properties": {
    "invoice_number": {
      "type": "string",
      "description": "The invoice number from the document",
      "x-idp-extraction-hint": "Look for 'Invoice #' or 'Invoice Number' near the top",
      "x-idp-confidence-threshold": 0.8,
      "x-idp-evaluation-method": "SEMANTIC",
      "x-idp-examples": ["INV-001", "INV-002"]
    }
  }
}
```

### 3. **Simplified UI**

- **Visual Editor**: Simple form-based field editor (no drag-and-drop
  complexity)
- **JSON Editor**: Direct JSON Schema editing with Monaco editor
- **Preview**: See what the schema looks like

## IDP Extension Fields

| Field                        | Type   | Description                                                   |
| ---------------------------- | ------ | ------------------------------------------------------------- |
| `x-idp-extraction-hint`      | string | Instructions for the AI on how to find this field             |
| `x-idp-confidence-threshold` | number | Minimum confidence score (0.0 to 1.0)                         |
| `x-idp-evaluation-method`    | enum   | Validation method: SEMANTIC, EXACT, FUZZY, NUMERIC_EXACT, LLM |
| `x-idp-examples`             | array  | Example values from real documents                            |

## Usage

### Visual Editor Tab

1. Click "Add Field" to create a new field
2. Select the field from the list
3. Edit properties in the right panel:
   - Basic: name, type, description, required
   - Type-specific: constraints based on field type
   - IDP Settings: extraction hints, evaluation method, confidence threshold
   - Advanced: JSON Schema standard properties

### JSON Schema Tab

- Direct editing of the JSON Schema
- Real-time validation
- Auto-syncs with visual editor

### Preview Tab

- See all fields and their configuration
- View IDP-specific settings
- Summary statistics

## Example Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "title": "Invoice",
  "description": "Invoice extraction schema",
  "properties": {
    "invoice_number": {
      "type": "string",
      "description": "Invoice number",
      "x-idp-extraction-hint": "Look for 'Invoice #' label",
      "x-idp-confidence-threshold": 0.9,
      "x-idp-evaluation-method": "EXACT",
      "x-idp-examples": ["INV-001", "INV-002"]
    },
    "invoice_date": {
      "type": "string",
      "format": "date",
      "description": "Invoice date",
      "x-idp-extraction-hint": "Date near invoice number",
      "x-idp-confidence-threshold": 0.8,
      "x-idp-evaluation-method": "SEMANTIC"
    },
    "total_amount": {
      "type": "number",
      "description": "Total amount",
      "minimum": 0,
      "x-idp-extraction-hint": "Look for 'Total' or 'Amount Due'",
      "x-idp-confidence-threshold": 0.85,
      "x-idp-evaluation-method": "NUMERIC_EXACT"
    },
    "line_items": {
      "type": "array",
      "description": "Invoice line items",
      "items": {
        "type": "object",
        "properties": {
          "description": {
            "type": "string",
            "x-idp-extraction-hint": "Item description in table"
          },
          "quantity": {
            "type": "integer",
            "minimum": 1
          },
          "unit_price": {
            "type": "number",
            "minimum": 0
          },
          "amount": {
            "type": "number",
            "minimum": 0
          }
        }
      },
      "x-idp-extraction-hint": "Extract table rows for each line item"
    }
  },
  "required": ["invoice_number", "invoice_date", "total_amount"]
}
```

## Migration from Old Format

The old IDP configuration format can be converted to JSON Schema with `x-idp-*`
extensions:

**Old Format:**

```yaml
classes:
  - name: Invoice
    attributes:
      - name: invoice_number
        attributeType: simple
        evaluation_method: EXACT
        confidence_threshold: 0.9
```

**New Format (JSON Schema):**

```json
{
  "type": "object",
  "properties": {
    "invoice_number": {
      "type": "string",
      "x-idp-evaluation-method": "EXACT",
      "x-idp-confidence-threshold": 0.9
    }
  }
}
```

## Benefits

1. **Standard Format**: JSON Schema is widely understood and supported
2. **Better Tooling**: Leverage existing JSON Schema validators and editors
3. **Simpler UI**: No complex drag-and-drop, just forms and direct editing
4. **Extensible**: Easy to add new IDP-specific fields via `x-idp-*` prefix
5. **Portable**: Can be used with any JSON Schema tooling
6. **Type Safety**: Clear type definitions for all fields
