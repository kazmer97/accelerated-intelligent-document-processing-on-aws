# Assessment: Frontend-Driven Extraction Configuration with Dynamic Pydantic Generation

## Executive Summary

The current extraction configuration system uses CloudFormation-defined schemas
that are rendered in the frontend. This assessment proposes inverting this flow:
empowering users to build extraction requirements through a flexible UI builder
that generates JSON schemas, which are then dynamically transformed into
Pydantic models on the backend for validation and processing.

## Current State Analysis

### Current Flow

```
CloudFormation YAML → JSON Schema → Frontend Form → Configuration → Backend Processing
```

### Limitations

1. **Static Schema Definition**: Schemas are hardcoded in CloudFormation
   templates
2. **Limited Flexibility**: Users cannot define custom extraction patterns
   without deployment
3. **No Runtime Validation**: Backend lacks strong typing for user-defined
   schemas
4. **Complex UI Code**: 1574-line FormView.jsx handles all rendering logic

## Proposed Solution: Frontend-Driven Schema with Dynamic Pydantic Generation

### New Architecture Flow

```
User Requirements → UI Builder → JSON Schema → Backend Pydantic Generation → Validation & Processing
        ↑                ↓              ↓                    ↓
        └── Visual Feedback ← Real-time Validation ← Type-safe Execution
```

### Core Components

#### 1. Frontend UI Builder System

##### A. Visual Schema Builder

```jsx
// Visual drag-and-drop schema builder
const ExtractionSchemaBuilder = () => {
  const [schema, setSchema] = useState({
    type: 'object',
    properties: {},
    required: [],
  });

  return (
    <SchemaBuilderLayout>
      <ComponentPalette>
        <DraggableField type="text" label="Text Field" />
        <DraggableField type="number" label="Numeric Field" />
        <DraggableField type="date" label="Date Field" />
        <DraggableField type="currency" label="Currency Field" />
        <DraggableField type="group" label="Field Group" />
        <DraggableField type="list" label="List/Table" />
        <DraggableField type="conditional" label="Conditional Field" />
      </ComponentPalette>

      <CanvasArea
        schema={schema}
        onDrop={handleFieldDrop}
        onFieldConfig={handleFieldConfiguration}
      />

      <PropertiesPanel
        selectedField={selectedField}
        onUpdate={updateFieldProperties}
      />
    </SchemaBuilderLayout>
  );
};
```

##### B. Field Configuration Components

```jsx
// Intelligent field configuration with extraction hints
const FieldConfigurator = ({ field, onUpdate }) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  return (
    <Container>
      <BasicProperties>
        <Input
          label="Field Name"
          value={field.name}
          onChange={(name) => onUpdate({ ...field, name })}
        />

        <RichTextEditor
          label="Extraction Instructions"
          value={field.description}
          placeholder="Describe how to identify this field in documents..."
          suggestions={getAISuggestions(field.type)}
        />

        <Select
          label="Data Type"
          value={field.type}
          options={[
            'string',
            'number',
            'date',
            'currency',
            'boolean',
            'array',
            'object',
          ]}
        />
      </BasicProperties>

      <ValidationRules>
        <ValidationBuilder field={field} onAddRule={handleAddValidation} />
      </ValidationRules>

      {showAdvanced && (
        <AdvancedOptions>
          <ExtractionHints field={field} documentSamples={uploadedSamples} />
          <ConfidenceSettings />
          <PostProcessingRules />
        </AdvancedOptions>
      )}
    </Container>
  );
};
```

##### C. Smart Schema Templates

```jsx
const SchemaTemplateLibrary = () => {
  const templates = [
    {
      name: 'Invoice Extraction',
      category: 'Financial',
      schema: invoiceSchema,
      preview: InvoicePreview,
    },
    {
      name: 'Bank Statement',
      category: 'Financial',
      schema: bankStatementSchema,
      preview: BankStatementPreview,
    },
    {
      name: 'Medical Record',
      category: 'Healthcare',
      schema: medicalRecordSchema,
      preview: MedicalRecordPreview,
    },
    // Custom user templates...
  ];

  return (
    <TemplateGallery>
      {templates.map((template) => (
        <TemplateCard
          key={template.name}
          template={template}
          onSelect={() => applyTemplate(template.schema)}
          onCustomize={() => openCustomizer(template)}
        />
      ))}
    </TemplateGallery>
  );
};
```

##### D. JSON Schema Generator

```javascript
class SchemaGenerator {
  generateJSONSchema(uiDefinition) {
    const schema = {
      $schema: 'http://json-schema.org/draft-07/schema#',
      type: 'object',
      properties: {},
      required: [],
      // Custom extensions for Pydantic generation
      'x-pydantic-config': {
        title: uiDefinition.name,
        use_enum_values: true,
        validate_assignment: true,
      },
    };

    // Convert UI fields to JSON Schema properties
    uiDefinition.fields.forEach((field) => {
      schema.properties[field.name] = this.fieldToSchemaProperty(field);

      if (field.required) {
        schema.required.push(field.name);
      }
    });

    return schema;
  }

  fieldToSchemaProperty(field) {
    const property = {
      type: field.type,
      description: field.description,
      // Pydantic-specific metadata
      'x-pydantic-field': {
        alias: field.displayName,
        validation: field.validationRules,
        extraction_hints: field.extractionHints,
        confidence_threshold: field.confidenceThreshold,
      },
    };

    // Add type-specific properties
    switch (field.type) {
      case 'string':
        if (field.pattern) property.pattern = field.pattern;
        if (field.enum) property.enum = field.enum;
        if (field.minLength) property.minLength = field.minLength;
        if (field.maxLength) property.maxLength = field.maxLength;
        break;

      case 'number':
        if (field.minimum) property.minimum = field.minimum;
        if (field.maximum) property.maximum = field.maximum;
        if (field.multipleOf) property.multipleOf = field.multipleOf;
        break;

      case 'array':
        property.items = this.fieldToSchemaProperty(field.itemDefinition);
        if (field.minItems) property.minItems = field.minItems;
        if (field.maxItems) property.maxItems = field.maxItems;
        break;

      case 'object':
        property.properties = {};
        field.nestedFields.forEach((nested) => {
          property.properties[nested.name] = this.fieldToSchemaProperty(nested);
        });
        break;
    }

    return property;
  }
}
```

#### 2. Backend Dynamic Pydantic Generation

##### A. JSON Schema to Pydantic Converter

```python
from typing import Type, Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, create_model, validator
from pydantic.fields import FieldInfo
import json

class DynamicPydanticGenerator:
    """Convert JSON Schema to Pydantic models dynamically"""

    def __init__(self):
        self.models_cache = {}
        self.field_validators = {}

    def generate_model_from_schema(
        self,
        schema: Dict[str, Any],
        model_name: str = "DynamicModel"
    ) -> Type[BaseModel]:
        """Generate Pydantic model from JSON Schema"""

        # Check cache
        schema_hash = self._hash_schema(schema)
        if schema_hash in self.models_cache:
            return self.models_cache[schema_hash]

        # Parse schema
        properties = schema.get('properties', {})
        required = schema.get('required', [])

        # Build field definitions
        field_definitions = {}
        validators = {}

        for field_name, field_schema in properties.items():
            # Create field with Pydantic Field
            field_type, field_info = self._create_field(
                field_name,
                field_schema,
                field_name in required
            )
            field_definitions[field_name] = (field_type, field_info)

            # Add custom validators if specified
            if 'x-pydantic-field' in field_schema:
                validators.update(
                    self._create_validators(field_name, field_schema['x-pydantic-field'])
                )

        # Add Pydantic config from schema
        config = self._create_config(schema.get('x-pydantic-config', {}))

        # Create the model dynamically
        dynamic_model = create_model(
            model_name,
            **field_definitions,
            __config__=config,
            __validators__=validators
        )

        # Cache the model
        self.models_cache[schema_hash] = dynamic_model

        return dynamic_model

    def _create_field(
        self,
        field_name: str,
        field_schema: Dict,
        is_required: bool
    ) -> tuple:
        """Create Pydantic field from JSON Schema property"""

        # Determine Python type
        python_type = self._get_python_type(field_schema)

        # Extract Pydantic-specific metadata
        pydantic_meta = field_schema.get('x-pydantic-field', {})

        # Build Field arguments
        field_args = {
            'description': field_schema.get('description', ''),
            'alias': pydantic_meta.get('alias', field_name),
        }

        # Add validation constraints
        if field_schema.get('type') == 'string':
            if 'minLength' in field_schema:
                field_args['min_length'] = field_schema['minLength']
            if 'maxLength' in field_schema:
                field_args['max_length'] = field_schema['maxLength']
            if 'pattern' in field_schema:
                field_args['regex'] = field_schema['pattern']

        elif field_schema.get('type') in ['number', 'integer']:
            if 'minimum' in field_schema:
                field_args['ge'] = field_schema['minimum']
            if 'maximum' in field_schema:
                field_args['le'] = field_schema['maximum']
            if 'multipleOf' in field_schema:
                field_args['multiple_of'] = field_schema['multipleOf']

        # Handle optional fields
        if not is_required:
            python_type = Optional[python_type]
            field_args['default'] = None

        return python_type, Field(**field_args)

    def _get_python_type(self, field_schema: Dict) -> Type:
        """Map JSON Schema type to Python type"""

        type_mapping = {
            'string': str,
            'number': float,
            'integer': int,
            'boolean': bool,
            'null': type(None)
        }

        json_type = field_schema.get('type')

        if json_type == 'array':
            item_type = self._get_python_type(field_schema.get('items', {}))
            return List[item_type]

        elif json_type == 'object':
            # Handle nested objects
            if 'properties' in field_schema:
                nested_model = self.generate_model_from_schema(
                    field_schema,
                    model_name=f"Nested_{id(field_schema)}"
                )
                return nested_model
            else:
                return Dict[str, Any]

        elif 'enum' in field_schema:
            # Create Enum type
            from enum import Enum
            enum_values = {f"VALUE_{i}": v for i, v in enumerate(field_schema['enum'])}
            return Enum(f"Enum_{id(field_schema)}", enum_values)

        return type_mapping.get(json_type, Any)

    def _create_validators(self, field_name: str, pydantic_meta: Dict) -> Dict:
        """Create custom validators from metadata"""
        validators = {}

        if 'validation' in pydantic_meta:
            for i, rule in enumerate(pydantic_meta['validation']):
                validator_name = f"validate_{field_name}_{i}"
                validators[validator_name] = validator(field_name, allow_reuse=True)(
                    self._create_validation_function(rule)
                )

        return validators

    def _create_validation_function(self, rule: Dict):
        """Create validation function from rule definition"""
        def validate(cls, v, values):
            # Implement custom validation logic based on rule
            if rule['type'] == 'custom_regex':
                import re
                if not re.match(rule['pattern'], str(v)):
                    raise ValueError(f"Value does not match pattern {rule['pattern']}")

            elif rule['type'] == 'business_rule':
                # Execute business rule validation
                exec_context = {'value': v, 'values': values}
                exec(rule['code'], exec_context)
                if not exec_context.get('valid', True):
                    raise ValueError(rule.get('message', 'Validation failed'))

            return v

        return validate

    def _create_config(self, config_dict: Dict) -> Type:
        """Create Pydantic Config class"""
        class Config:
            arbitrary_types_allowed = True
            use_enum_values = config_dict.get('use_enum_values', True)
            validate_assignment = config_dict.get('validate_assignment', True)
            extra = config_dict.get('extra', 'forbid')

        return Config

    def _hash_schema(self, schema: Dict) -> str:
        """Create hash of schema for caching"""
        import hashlib
        schema_json = json.dumps(schema, sort_keys=True)
        return hashlib.md5(schema_json.encode()).hexdigest()
```

##### B. Extraction Service Integration

```python
class DynamicExtractionService:
    """Service for processing documents with dynamic schemas"""

    def __init__(self, schema_generator: DynamicPydanticGenerator):
        self.schema_generator = schema_generator
        self.compiled_models = {}

    async def extract_with_schema(
        self,
        document: str,
        schema: Dict[str, Any],
        extraction_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract data from document using dynamic schema"""

        # Generate Pydantic model from schema
        model_class = self.schema_generator.generate_model_from_schema(
            schema,
            model_name=f"ExtractionModel_{schema.get('title', 'Dynamic')}"
        )

        # Prepare extraction prompt with schema information
        prompt = self._build_extraction_prompt(document, schema, model_class)

        # Call LLM for extraction
        raw_extraction = await self._llm_extract(prompt, extraction_config)

        # Validate with Pydantic model
        try:
            validated_data = model_class(**raw_extraction)
            return {
                'success': True,
                'data': validated_data.dict(),
                'validation_errors': []
            }
        except ValidationError as e:
            return {
                'success': False,
                'data': raw_extraction,
                'validation_errors': e.errors()
            }

    def _build_extraction_prompt(
        self,
        document: str,
        schema: Dict,
        model_class: Type[BaseModel]
    ) -> str:
        """Build extraction prompt with schema context"""

        # Generate field descriptions from Pydantic model
        field_descriptions = []
        for field_name, field_info in model_class.__fields__.items():
            extraction_hints = schema.get('properties', {}).get(
                field_name, {}
            ).get('x-pydantic-field', {}).get('extraction_hints', '')

            field_descriptions.append(f"""
            - {field_name}: {field_info.field_info.description}
              Type: {field_info.type_}
              Extraction hints: {extraction_hints}
            """)

        prompt = f"""
        Extract the following information from the document:

        {chr(10).join(field_descriptions)}

        Document:
        {document}

        Return the extracted data as a JSON object matching the schema.
        """

        return prompt
```

##### C. Validation and Feedback Service

```python
class SchemaValidationService:
    """Real-time validation and feedback for frontend schemas"""

    async def validate_schema(self, schema: Dict) -> Dict:
        """Validate schema and provide feedback"""

        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        try:
            # Try to generate Pydantic model
            generator = DynamicPydanticGenerator()
            model = generator.generate_model_from_schema(schema)

            # Test with sample data
            test_results = self._test_schema_with_samples(model, schema)
            validation_result['test_results'] = test_results

            # Analyze schema for improvements
            suggestions = self._analyze_schema_quality(schema)
            validation_result['suggestions'] = suggestions

        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(str(e))

        return validation_result

    def _analyze_schema_quality(self, schema: Dict) -> List[str]:
        """Provide suggestions for schema improvement"""
        suggestions = []

        # Check for missing descriptions
        for prop_name, prop_schema in schema.get('properties', {}).items():
            if not prop_schema.get('description'):
                suggestions.append(
                    f"Add description for field '{prop_name}' to improve extraction accuracy"
                )

            # Check for validation rules
            if prop_schema.get('type') == 'string' and not any([
                'pattern' in prop_schema,
                'enum' in prop_schema,
                'minLength' in prop_schema
            ]):
                suggestions.append(
                    f"Consider adding validation rules for string field '{prop_name}'"
                )

        return suggestions
```

#### 3. Real-time Communication Layer

##### WebSocket Handler for Live Validation

```python
class SchemaWebSocketHandler:
    """WebSocket handler for real-time schema validation"""

    async def handle_connection(self, websocket, path):
        generator = DynamicPydanticGenerator()

        async for message in websocket:
            data = json.loads(message)

            if data['type'] == 'validate_schema':
                # Validate schema in real-time
                result = await self._validate_schema(
                    data['schema'],
                    generator
                )
                await websocket.send(json.dumps({
                    'type': 'validation_result',
                    'result': result
                }))

            elif data['type'] == 'test_extraction':
                # Test extraction with sample document
                result = await self._test_extraction(
                    data['schema'],
                    data['sample_document'],
                    generator
                )
                await websocket.send(json.dumps({
                    'type': 'extraction_test_result',
                    'result': result
                }))
```

### Implementation Benefits

#### 1. **User Empowerment**

- Non-technical users can define extraction schemas
- Visual builder with drag-and-drop interface
- Real-time validation feedback
- Template library for common use cases

#### 2. **Dynamic Flexibility**

- No deployment needed for schema changes
- Runtime model generation
- Custom validation rules per field
- Conditional field logic support

#### 3. **Type Safety**

- Dynamic Pydantic models provide runtime type checking
- Automatic validation of extracted data
- Clear error messages for validation failures
- IDE support for generated models

#### 4. **Scalability**

- Schema caching for performance
- Async processing support
- Modular architecture
- Easy to extend with new field types

### Migration Path

#### Phase 1: Frontend UI Builder (3-4 weeks)

1. Implement visual schema builder
2. Create field configuration components
3. Build JSON Schema generator
4. Add template library

#### Phase 2: Backend Pydantic Generator (2-3 weeks)

1. Implement JSON Schema to Pydantic converter
2. Add validation service
3. Create model caching layer
4. Build extraction service integration

#### Phase 3: Real-time Validation (1-2 weeks)

1. Set up WebSocket infrastructure
2. Implement live validation
3. Add feedback mechanisms
4. Create testing interface

#### Phase 4: Integration (2 weeks)

1. Connect frontend to backend services
2. Update existing extraction pipeline
3. Migrate existing configurations
4. Performance optimization

### Example Workflow

```javascript
// 1. User builds schema in UI
const userSchema = {
  title: 'Invoice Extraction',
  type: 'object',
  properties: {
    invoiceNumber: {
      type: 'string',
      description: 'The invoice number, usually at the top right',
      pattern: '^INV-[0-9]{6}$',
      'x-pydantic-field': {
        extraction_hints: "Look for 'Invoice #' or 'Invoice Number'",
        confidence_threshold: 0.95,
      },
    },
    totalAmount: {
      type: 'number',
      description: 'Total amount including tax',
      minimum: 0,
      'x-pydantic-field': {
        extraction_hints: "Usually labeled 'Total' or 'Amount Due'",
        validation: [
          {
            type: 'business_rule',
            code: 'valid = value > 0',
            message: 'Total amount must be positive',
          },
        ],
      },
    },
    lineItems: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          description: { type: 'string' },
          quantity: { type: 'number' },
          unitPrice: { type: 'number' },
          total: { type: 'number' },
        },
      },
    },
  },
  required: ['invoiceNumber', 'totalAmount'],
};

// 2. Frontend sends to backend
await api.saveExtractionSchema(userSchema);

// 3. Backend generates Pydantic model dynamically
// 4. Extraction uses the dynamic model for validation
```

### Performance Considerations

1. **Caching Strategy**

   - Cache generated Pydantic models
   - Store compiled schemas in Redis
   - Implement schema versioning

2. **Optimization**

   - Lazy loading of complex validators
   - Batch validation for multiple documents
   - Async processing pipeline

3. **Monitoring**
   - Track schema generation time
   - Monitor validation performance
   - Log extraction success rates

### Conclusion

This frontend-driven approach with dynamic Pydantic generation provides the
perfect balance of user flexibility and backend type safety. Users can visually
design extraction schemas that are automatically transformed into robust,
validated Pydantic models at runtime. This eliminates the need for deployment
cycles while maintaining strong validation and type checking throughout the
extraction pipeline.

The JSON Schema serves as the perfect interoperation layer, being both
human-readable and machine-processable, while the dynamic Pydantic generation
ensures that all the benefits of typed Python models are retained without
sacrificing flexibility.
