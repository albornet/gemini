from typing import Type, Any, Union, List
from pydantic import BaseModel, Field, create_model
from vllm.sampling_params import GuidedDecodingParams
from src.utils.run_utils import load_config

cfg = load_config()


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
) -> Union[dict[str, Any], GuidedDecodingParams, Type[BaseModel]]:
    """ Dynamically creates a pydantic BaseModel class from yaml configuration
    """
    # Extract Pydantic style output schema from the configuration
    output_schema = create_pydantic_model_from_schema_dict(
        schema_dict=cfg["output_schema"],
        model_name=cfg["output_schema_name"],
    )
    
    # Return an output guide corresponding to the backend used for LLM inference
    match inference_backend:
        case "llama-cpp":
            return output_schema.model_json_schema()
        case "vllm":
            json_schema = output_schema.model_json_schema()
            return GuidedDecodingParams(
                json=json_schema,
                backend="xgrammar:no-fallback",
            )
        case "hf":
            return output_schema  # not sure how to handle the classif HugginfFace case
        case _:
            raise ValueError(f"Unsupported inference backend: {inference_backend}")


if __name__ == "__main__":
    output_schema = create_pydantic_model_from_schema_dict(
        schema_dict=cfg["output_schema"],
        model_name=cfg["output_schema_name"],
    )
    import ipdb; ipdb.set_trace()