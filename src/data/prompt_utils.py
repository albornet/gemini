from typing import Union, Annotated, Any
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm.sampling_params import GuidedDecodingParams


def build_prompt(input_text: str) -> list[dict[str, str]]:
    """ Build a message list for prompting a large language model for the task at hand

    Args:
        input_text (str): core input text for the current sample

    Returns:
        list[dict[str, str]]: messages to prompt a large language model
    """
    task_description = (
        "Tu es un neuro-chirurgien et ta tâche est d’identifier le score sur l’Échelle de Rankin Modifiée (mRS) du patient"
        "en fonction de son état après sa sortie, tel qu'il est décrit dans la lettre de sortie rédigée par un autre médecin."
    )
    mrs_description = (
        "L'Échelle de Rankin Modifiée (mRS) est utilisée pour mesurer le degré d'incapacité chez les patients ayant subi un accident vasculaire cérébral (AVC) :\n"
        "0 : Aucun symptôme\n"
        "1 : Aucune incapacité significative malgré des symptômes ; capable d'effectuer toutes les tâches et activités habituelles\n"
        "2 : Légère incapacité ; incapable d'effectuer toutes les activités antérieures, mais capable de s'occuper de ses propres affaires sans assistance\n"
        "3 : Incapacité modérée ; nécessitant une certaine aide, mais capable de marcher sans assistance\n"
        "4 : Incapacité modérément sévère ; incapable de marcher sans assistance et incapable de s'occuper de ses besoins corporels sans assistance\n"
        "5 : Incapacité sévère ; alité, incontinent et nécessitant des soins infirmiers constants et une attention continue\n"
        "6 : Décédé"
    )
    output_specs = (
        "Ta réponse doit être JSON conforme au format suivant :\n\n"
        "{\n"
        '  "reasoning": "<comment tu as déterminé le mRS>",\n'
        '  "mRS": <nombre entre 0 et 6>\n'
        "}\n\n"
    )
    system_prompt = f"{task_description}\n{mrs_description}\n{output_specs}"  # f"{task_description}\n{output_specs}"
    user_prompt = (
        "Voici la lettre de sortie du patient:\n"
        "DEBUT DE LA LETTRE DE SORTIE :\n"
        f"{input_text}\n"
        "FIN DE LA LETTRE DE SORTIE\n"
    )
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    return messages


def build_prompts(
    sample: dict[str, str],
    tokenizer: AutoTokenizer,
) -> dict[str, Union[list, str]]:
    """ Take in a data sample, build huggingface style's messages,
        add tokenize associated prompt
    """
    messages = build_prompt(sample["input_text"])
    prompt = None
    if tokenizer is not None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return {"messages": messages, "prompt": prompt}


def get_output_guide(
    inference_backend: str,
    max_generated_tokens: int=512,
) -> dict[str, Any]:
    """ Define a formatted output schema to constraint the LLM when generating tokens
    """
    class PatientInfoSchema(BaseModel):
        reasoning: str = Field(
            ...,
            max_length=int(max(max_generated_tokens * 0.9, max_generated_tokens - 100)),
            description="Le raisonnement du modèle pour déterminer le score mRS.",
            example=(
                "Le patient présente une faiblesse du côté gauche et nécessite"
                "une aide pour la marche, ce qui correspond à un score mRS de 3."
            ),
        )
        mRS: Annotated[int, Field(description="Échelle de Rankin Modifiée (0-6)", ge=0, le=6, example=3, title="mRS")]
        # visit_date: str = Field(..., description="Date au format jj/mm/aaaa")

    # Return output guide suited for the inference backend
    match inference_backend:

        case "llama-cpp":
            return PatientInfoSchema.model_json_schema()
        
        case "vllm":
            json_schema = PatientInfoSchema.model_json_schema()
            return GuidedDecodingParams(json=json_schema, backend="lm-format-enforcer") 
        
        case "hf":
            return PatientInfoSchema
        
        case _:
            raise ValueError(f"Unsupported inference backend: {inference_backend}")
