# Multi-Model Schema Builder

## Overview

The Multi-Model Schema Builder supports defining multiple reusable models (like
Pydantic classes) that can reference each other. This is essential for complex
document structures with nested entities.

## Key Features

### 1. **Multiple Models/Definitions**

- Create multiple models in a single schema
- Each model is a reusable type definition
- Models stored in JSON Schema `definitions` section

### 2. **Model References**

- Fields can reference other models using `$ref`
- Array fields can contain model instances
- Supports nested structures of any depth

### 3. **Main Model**

- One model is designated as the "main" entry point
- Marked with blue "Main" badge
- Set via `$ref` at root level

## JSON Schema Structure

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "definitions": {
    "Invoice": {
      "type": "object",
      "title": "Invoice",
      "description": "Main invoice document",
      "properties": {
        "invoice_number": {
          "type": "string"
        },
        "vendor": {
          "$ref": "#/definitions/Vendor"
        },
        "line_items": {
          "type": "array",
          "items": {
            "$ref": "#/definitions/LineItem"
          }
        }
      },
      "required": ["invoice_number"]
    },
    "Vendor": {
      "type": "object",
      "title": "Vendor",
      "description": "Vendor information",
      "properties": {
        "name": { "type": "string" },
        "address": { "$ref": "#/definitions/Address" }
      }
    },
    "Address": {
      "type": "object",
      "title": "Address",
      "properties": {
        "street": { "type": "string" },
        "city": { "type": "string" },
        "zip_code": { "type": "string" }
      }
    },
    "LineItem": {
      "type": "object",
      "title": "Line Item",
      "properties": {
        "description": { "type": "string" },
        "quantity": { "type": "integer" },
        "unit_price": { "type": "number" },
        "amount": { "type": "number" }
      }
    }
  },
  "$ref": "#/definitions/Invoice"
}
```

## UI Features

### **Model List (Left Panel)**

- View all defined models
- Add new models
- Select model to edit
- Set main model
- Delete models (can't delete last one)
- Blue badge shows main model

### **Field List (Middle Panel)**

- Shows fields for selected model
- Add/edit/delete fields
- Required fields marked with \*
- Reference fields marked with 🔗 icon
- Shows referenced model name

### **Field Editor (Right Panel)**

- Edit field properties
- Set field type (including Reference type)
- For references: select target model from dropdown
- For arrays: choose item type or reference model
- Configure IDP extraction settings

## Usage Examples

### Example 1: Invoice with Line Items

**Models:**

1. Invoice (main)
2. LineItem

**Invoice fields:**

- `invoice_number`: string
- `line_items`: array of LineItem references

**LineItem fields:**

- `description`: string
- `quantity`: integer
- `amount`: number

### Example 2: Healthcare Document

**Models:**

1. MedicalRecord (main)
2. Patient
3. Provider
4. Diagnosis
5. Medication

**Relationships:**

- MedicalRecord → Patient (single reference)
- MedicalRecord → Provider (single reference)
- MedicalRecord → Diagnosis (array of references)
- MedicalRecord → Medication (array of references)

### Example 3: Lending Package

**Models:**

1. LendingPackage (main)
2. Applicant
3. EmploymentHistory
4. FinancialStatement
5. CreditReport

**Nested References:**

```
LendingPackage
  ├─ applicants: [Applicant]
  │   └─ employment_history: [EmploymentHistory]
  ├─ financial_statements: [FinancialStatement]
  └─ credit_reports: [CreditReport]
```

## Creating a Multi-Model Schema

### Step 1: Add Models

1. Click "Add Model" in Models panel
2. Enter model name (e.g., "LineItem")
3. Add description
4. Click "Create Model"

### Step 2: Add Fields to Models

1. Select a model from the list
2. Click "Add Field" in Fields panel
3. Configure field properties
4. For references:
   - Select "Reference (link to another model)" as type
   - Choose target model from dropdown
5. For arrays of models:
   - Select "Array (list)" as type
   - In "Or Reference Model", select the model type

### Step 3: Set Main Model

1. Click "⋮" menu on a model
2. Select "Set as Main Model"
3. Model gets blue "Main" badge

### Step 4: Save

1. Review in JSON Schema tab
2. Check Preview tab
3. Click "Save Schema"

## Benefits Over Single Model

| Single Model          | Multi-Model                 |
| --------------------- | --------------------------- |
| Flat structure        | Hierarchical structure      |
| Repeated field groups | Reusable definitions        |
| Hard to maintain      | Easy to update shared types |
| Limited nesting       | Unlimited nesting           |
| Not DRY               | DRY principle               |

## Pydantic Equivalence

This multi-model schema is equivalent to Pydantic models:

**Multi-Model JSON Schema:**

```json
{
  "definitions": {
    "LineItem": {
      "properties": {
        "description": { "type": "string" },
        "amount": { "type": "number" }
      }
    },
    "Invoice": {
      "properties": {
        "line_items": {
          "type": "array",
          "items": { "$ref": "#/definitions/LineItem" }
        }
      }
    }
  }
}
```

**Equivalent Pydantic:**

```python
from pydantic import BaseModel

class LineItem(BaseModel):
    description: str
    amount: float

class Invoice(BaseModel):
    line_items: list[LineItem]
```

## Backend Integration

When this schema is sent to the backend:

1. **Parser** recognizes `definitions` structure
2. **Generator** creates Pydantic models for each definition
3. **References** are resolved to Python class references
4. **Validation** uses Pydantic's nested validation
5. **Extraction** maintains relationships between entities

## Best Practices

1. **Model Naming**: Use PascalCase (e.g., LineItem, not line_item)
2. **Granularity**: Create separate models for reusable entities
3. **Main Model**: Set the top-level document model as main
4. **Descriptions**: Add descriptions to help AI extraction
5. **IDP Settings**: Configure extraction hints on each field
6. **Testing**: Use Preview tab to verify structure

## Migration from Single Model

To convert existing single-model schemas:

1. Keep main model structure in "Document" definition
2. Extract repeated nested objects into separate models
3. Replace nested properties with `$ref` to new models
4. Update arrays to use `items: {$ref: "..."}`
5. Set Document as main model

Example:

```json
// Before (single model)
{
  "properties": {
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"}
        }
      }
    }
  }
}

// After (multi-model)
{
  "definitions": {
    "Item": {
      "properties": {
        "name": {"type": "string"}
      }
    },
    "Document": {
      "properties": {
        "items": {
          "type": "array",
          "items": {"$ref": "#/definitions/Item"}
        }
      }
    }
  },
  "$ref": "#/definitions/Document"
}
```

## Troubleshooting

### Issue: Can't delete model

**Solution**: You cannot delete the last model. Create another model first.

### Issue: Reference not showing in dropdown

**Solution**: Make sure you've created the target model first.

### Issue: Circular references

**Solution**: JSON Schema and Pydantic support circular references, but use with
caution.

### Issue: Array of references not working

**Solution**: Make sure to use the "Or Reference Model" dropdown in array
configuration, not the basic "Item Type".
