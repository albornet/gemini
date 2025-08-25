import re
import json
import json5
from typing import Type, Any, Union, List
from pydantic import BaseModel, ValidationError, Field, create_model
from pydantic_core import PydanticUndefinedType
from vllm.sampling_params import GuidedDecodingParams


# Mappings from json schema keywords to pydantic field arguments
TYPE_MAPPING = {"string": str, "integer": int, "number": float, "boolean": bool}
CONSTRAINT_MAPPING = {

    # No-constraint keys
    "default": None,
    "description": None,
    "title": None,

    # String
    "maxLength": "max_length",
    "minLength": "min_length",
    "pattern": "pattern",

    # Numeric
    "minimum": "ge",
    "maximum": "le",
    "exclusiveMinimum": "gt",
    "exclusiveMaximum": "lt",
    "multipleOf": "multiple_of",

    # Array
    "minItems": "min_length",  # pydantic uses min_length for lists
    "maxItems": "max_length",  # pydantic uses max_length for lists

    # Need others?
    # ...

}


def parse_schema_field(
    field_name: str,
    properties: dict[str, Any],
) -> tuple[Type, dict[str, Any]]:
    """ Parse a schema property to Pydantic type and constraints
    """
    # Extract field type
    type_str = properties.get("type")
    if not type_str:
        raise ValueError(f"Field '{field_name}' must have a 'type'.")

    # Map json-style type to python type (this handles most "basic" types)
    if type_str in TYPE_MAPPING:
        pydantic_type = TYPE_MAPPING[type_str]

    # Special case for lists ("array" in json-style)
    elif type_str == "array":
        items_schema = properties.get("items")
        if not isinstance(items_schema, dict) or "type" not in items_schema:
            raise ValueError(f"Array '{field_name}' needs 'items' with a 'type'.")
        
        # Identify the type of the items in the list
        item_pydantic_type, _ = parse_schema_field(f"{field_name}_items", items_schema.copy())
        pydantic_type = List[item_pydantic_type]

    else:
        raise ValueError(f"Unsupported type '{type_str}' for '{field_name}'.")
    
    # Map any identified json-style constraints to pydantic constraints
    helpers = {}
    constraints = {}
    for schema_key, pydantic_key in CONSTRAINT_MAPPING.items():
        if schema_key in properties and pydantic_key is not None:
            constraints[pydantic_key] = properties[schema_key]
        elif schema_key in properties:
            if pydantic_key is None:
                helpers[schema_key] = properties[schema_key]
            else:
                raise ValueError(f"Unsupported constraint '{schema_key}' for property '{field_name}'.")

    # Combine all arguments in a single dictionary
    field_args = {**helpers, **constraints}

    # TODO: ensure extracted constraints are applicable to the determined pydantic_type
    return pydantic_type, field_args


def create_pydantic_model_from_schema_dict(
    schema_dict: dict[str, Any],
    model_name: str,
) -> Type[BaseModel]:
    """ Create a pydantic model from a schema dictionary
    """
    pydantic_fields = {}
    for field_name, field_properties in schema_dict.items():
        if not isinstance(field_properties, dict):
            raise ValueError(f"Properties for '{field_name}' must be a dict.")
        
        field_type, field_args = parse_schema_field(field_name, field_properties)
        pydantic_fields[field_name] = (field_type, Field(**field_args))

    return create_model(model_name, **pydantic_fields, __base__=BaseModel)
    

def create_output_guide(
    inference_backend: str,
    output_schema_dict: dict[str, Any],
    output_schema_name: str,
) -> Union[dict[str, Any], GuidedDecodingParams, Type[BaseModel]]:
    """ Dynamically creates a pydantic BaseModel class from yaml configuration
    """
    # Extract Pydantic style output schema from the configuration
    output_schema = create_pydantic_model_from_schema_dict(
        schema_dict=output_schema_dict,
        model_name=output_schema_name,
    )

    # Return an output guide corresponding to the backend used for LLM inference
    match inference_backend:
        case "llama-cpp":
            return output_schema.model_json_schema()
        case "vllm":
            json_schema = output_schema.model_json_schema()
            return GuidedDecodingParams(json=json_schema)
        case "hf":
            return output_schema  # not sure how to handle the classif HugginfFace case
        case _:
            raise ValueError(f"Unsupported inference backend: {inference_backend}")


def extract_structured_output(
    sample: dict[str, Any],
    output_schema_model: Type[BaseModel],
) -> dict[str, Any]:
    """
    Extracts structured output from raw model output using a multi-stage
    lenient parsing strategy.

    Args:
        sample (dict[str, Any]): The sample containing model output.
        output_schema_model (Type[BaseModel]): The Pydantic model for validation.
    
    Returns:
        dict[str, Any]: Structured and validated output from the LLM.
    """
    raw_output = sample.get("output_text")
    if not isinstance(raw_output, str) or not raw_output.strip():
        # Handle cases where output is missing or empty
        print("Warning: 'output_text' is missing or empty. Returning default values.")
        return _get_default_values(output_schema_model)

    # Direct pydantic parse
    try:
        validated_output = output_schema_model.model_validate_json(raw_output.strip())
        print("Success: Direct parsing successful.")
        return validated_output.model_dump()
    except (ValidationError, json.JSONDecodeError):
        pass  # silently fail and move to the next stage

    # Isolate JSON block and look for markdown code fences
    json_candidate = raw_output
    match = re.search(r"```(json)?\s*({.*})\s*```", raw_output, re.DOTALL)
    if match:
        json_candidate = match.group(2)
    else:
        # Fallback to finding the first '{' and last '}'
        try:
            start = raw_output.index("{")
            end = raw_output.rindex("}") + 1
            json_candidate = raw_output[start:end]
        except ValueError:
            pass  # no JSON object found
    try:
        validated_output = output_schema_model.model_validate_json(json_candidate)
        print("Success: Isolated and parsed JSON block.")
        return validated_output.model_dump()
    except (ValidationError, json.JSONDecodeError):
        pass

    # Parse with json5 library (more lenient with trailing commas, comments, etc.)
    try:
        data = json5.loads(json_candidate)
        validated_output = output_schema_model.model_validate(data)
        print("Success: Parsed with lenient json5 library.")
        return validated_output.model_dump()
    except (ValidationError, Exception):  # json5 might raise other errors
        pass

    # Attempt to repair truncation
    try:
        repaired_json = _repair_truncated_json(json_candidate)
        validated_output = output_schema_model.model_validate_json(repaired_json)
        print("Success: Repaired truncated JSON and parsed.")
        return validated_output.model_dump()
    except (ValidationError, json.JSONDecodeError):
        pass

    # Field-by-field regex extraction
    print("Warning: All parsing methods failed. Attempting field-by-field regex extraction.")
    extracted_data = {}
    for field_name, field_info in output_schema_model.model_fields.items():
        field_type = field_info.annotation
        value = _extract_field_with_regex(json_candidate, field_name, field_type)
        if value is not None:
            extracted_data[field_name] = value

    if extracted_data:
        print(f"Success: Extracted partial data with regex: {list(extracted_data.keys())}")
        
        # Merge extracted fields with default ones for missing fields
        defaults = _get_default_values(output_schema_model)
        defaults.update(extracted_data)

        # Final validation pass on the cobbled-together data
        try:
            final_model = output_schema_model.model_validate(defaults)
            return final_model.model_dump()
        except ValidationError:
            # If validation still fails, return what we have plus defaults
            return defaults

    # Final Fallback (return only default values)
    print("Error: Could not extract any fields. Returning default values.")
    return _get_default_values(output_schema_model)


def _repair_truncated_json(s: str) -> str:
    """
    Append missing brackets/braces to a potentially truncated JSON string
    """
    open_braces = s.count('{')
    close_braces = s.count('}')
    open_brackets = s.count('[')
    close_brackets = s.count(']')

    s += '}' * (open_braces - close_braces)
    s += ']' * (open_brackets - close_brackets)
    return s


def _extract_field_with_regex(
    text: str,
    field_name: str,
    field_type: Type,
) -> Any | None:
    """
    Extracts a single field value using a type-aware regex pattern
    """
    # Pattern for string: "field_name"\s*:\s*"<value>"
    if field_type == str:

        # Captures content inside double quotes, handling escaped quotes
        pattern = rf'"{field_name}"\s*:\s*"((?:\\"|[^"])*)"'
        match = re.search(pattern, text)
        return match.group(1).replace('\\"', '"') if match else None

    # Pattern for number (int/float): "field_name"\s*:\s*<value>
    if field_type in (int, float):
        pattern = rf'"{field_name}"\s*:\s*(-?\d+\.?\d*)'
        match = re.search(pattern, text)
        return field_type(match.group(1)) if match else None

    # Pattern for boolean: "field_name"\s*:\s*<value>
    if field_type == bool:
        pattern = rf'"{field_name}"\s*:\s*(true|false)'
        match = re.search(pattern, text)
        if match:
            return match.group(1) == 'true'
        return None
    
    return None


def _get_default_values(model: Type[BaseModel]) -> dict[str, Any]:
    """
    Builds a dictionary of default values from a Pydantic model
    """
    defaults = {}
    for name, field in model.model_fields.items():
        if field.default_factory:
            defaults[name] = field.default_factory()
        elif not isinstance(field.default, PydanticUndefinedType):
            defaults[name] = field.default
        else:
            defaults[name] = None  # for required fields with no default
    
    return defaults
