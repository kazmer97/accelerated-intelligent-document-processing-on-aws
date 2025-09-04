"""
Agentic IDP implementation using Strands agents with tool-based structured output.

This module implements structured data extraction using Strands agents and tools,
recreating the structured_output_async functionality from ai-tools-registry using
tool-based approach with dynamic tool creation based on Pydantic models.
"""

import asyncio
import io
import json
import logging
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

import jsonpatch
from PIL import Image, ImageEnhance, ImageOps
from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.types.content import ContentBlock, Message
from strands.types.media import ImageContent, ImageSource

logger = logging.getLogger(__name__)

TargetModel = TypeVar("TargetModel", bound=BaseModel)


class BedrockUsage(TypedDict, total=False):
    """Token usage information from Bedrock response."""

    inputTokens: int
    outputTokens: int
    totalTokens: int
    cacheReadInputTokens: int
    cacheWriteInputTokens: int


class BedrockMessageContent(TypedDict):
    """Content item in a Bedrock message."""

    text: Optional[str]


class BedrockMessage(TypedDict):
    """Message structure in Bedrock response."""

    role: str
    content: List[BedrockMessageContent]


class BedrockOutput(TypedDict):
    """Output structure in Bedrock response."""

    message: BedrockMessage


class BedrockResponse(TypedDict, total=False):
    """Raw response from Bedrock converse API."""

    output: BedrockOutput
    usage: BedrockUsage
    stopReason: Optional[str]
    metrics: Optional[Dict[str, Any]]


class BedrockInvokeModelResponse(TypedDict):
    """
    Complete response structure from bedrock.invoke_model method.

    This represents the structure returned by:
    response_with_metering = bedrock.invoke_model(...)

    The response contains both the raw Bedrock API response and
    metering information with usage statistics.
    """

    response: BedrockResponse
    metering: Dict[str, BedrockUsage]  # Key format: "{context}/bedrock/{model_id}"


# Data Models for structured extraction
class BoolResponseModel(BaseModel):
    """Model for boolean validation responses."""

    valid_result: bool
    description: str = Field(..., description="explanation of the decision")


class JsonPatchModel(BaseModel):
    """Model for JSON patch operations."""

    patches: List[Dict[str, Any]] = Field(
        ...,
        description="JSON patch operations to apply. Each patch should follow RFC 6902 format with 'op', 'path', and optionally 'value' keys.",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of what these patches are intended to fix or update",
    )


def apply_patches_to_data(
    existing_data: Dict[str, Any],
    patches: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Apply JSON patches to existing data and validate the result.

    Args:
        existing_data: The current structured data to patch
        patches: List of JSON patch operations

    Returns:
        Patched and validated data
    """
    if not patches:
        return existing_data

    patch = jsonpatch.JsonPatch(patches)
    patched_dict = patch.apply(existing_data)

    return patched_dict


def create_dynamic_extraction_tool_and_patch_tool(model_class: Type[TargetModel]):
    """
    Create a dynamic tool function that extracts data according to a Pydantic model.

    This follows the pattern from ai-tools-registry where the tool's input schema
    is dynamically generated from the Pydantic model, ensuring the LLM knows exactly
    what structure to provide.

    Args:
        model_class: The Pydantic model class to use for extraction

    Returns:
        A tool-decorated function that validates against the model
    """

    @tool
    def _extraction_tool(
        extraction: dict,  # pyright: ignore[reportInvalidTypeForm]
        agent: Agent,  # pyright: ignore[reportInvalidTypeForm]
    ) -> dict:  # pyright: ignore[reportInvalidTypeForm]
        logger.debug(f"🔧 _extraction_tool extraction data: {extraction}")

        # Validate the extraction is the correct model type
        agent.state.set(key="current_extraction", value=extraction)
        extraction_p_model = model_class(**extraction)
        # Store in agent state
        extraction_dict = extraction_p_model.model_dump()
        agent.state.set(key="current_extraction", value=extraction_dict)

        return extraction_dict

    @tool
    def apply_json_patches(
        patches: List[Dict[str, Any]],
        reasoning: str,
        agent: Agent,
    ) -> Dict[str, Any]:
        """
        Apply JSON patches to fix or update the extracted data.

        Args:
            patches: List of JSON patch operations (RFC 6902 format)
            reasoning: Explanation of what the patches fix
        """
        current_data: Dict | None = agent.state.get("current_extraction")

        if not current_data:
            return {"error": "No current extraction to patch"}

        logger.debug(f"🔧 Applying {len(patches)} patches: {reasoning}")

        patched_data = apply_patches_to_data(current_data, patches)
        # Reconstruct the model instance if we only have dict
        agent.state.set("current_extraction", patched_data)

        validated_patched_data = model_class(**patched_data)

        # Update state
        agent.state.set(key="current_extraction", value=validated_patched_data)

        logger.debug(f"✅ Successfully applied {len(patches)} patches")
        return {
            "status": "success",
            "data": validated_patched_data.model_dump(),
            "patches_applied": len(patches),
        }

    _extraction_tool.__doc__ = f"""
        Use this tool to return the requested data extraction
        This tool needs to be Successfully invoked before the patch tool can be used.
        required extraction schema is: {model_class.model_json_schema()}"""
    return _extraction_tool, apply_json_patches


@tool
def enhance_image(
    image_index: int,
    enhancement_type: Literal["brightness", "contrast", "sharpen", "grayscale"],
    agent: Agent,
    factor: float = 1.5,
) -> str:
    """
    Apply image enhancement for better extraction accuracy.

    Args:
        image_index: Index of the image to enhance (0-based)
        enhancement_type: Type of enhancement (brightness, contrast, sharpen, grayscale)
        factor: Enhancement factor (for brightness/contrast)
    """
    images = agent.state.get("images")

    if image_index not in images:
        return f"Image index {image_index} not found"

    image = images[image_index]

    if enhancement_type == "brightness":
        enhancer = ImageEnhance.Brightness(image)
        enhanced = enhancer.enhance(factor)
    elif enhancement_type == "contrast":
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(factor)
    elif enhancement_type == "sharpen":
        enhancer = ImageEnhance.Sharpness(image)
        enhanced = enhancer.enhance(factor)
    elif enhancement_type == "grayscale":
        enhanced = ImageOps.grayscale(image)

    # Update the image in context
    images[image_index] = enhanced
    agent.state.set(key="images", value=images)

    return f"Applied {enhancement_type} to image {image_index}"


@tool
def load_image(image_index: int, agent: Agent) -> Image.Image:
    """
    This tool allows you to retrieve an image to view.
    """
    images = agent.state.get("image_index")
    if image_index not in images:
        raise ValueError(f"Image indes: {image_index}, does not exist")

    return images[image_index]


# System prompt matching the original
SYSTEM_PROMPT = """
You are a useful assistant that helps turn unstructured data into structured data using the provided tools.

EXTRACTION APPROACH:
1. Use the _extraction_tool for fresh data extraction
2. When updating existing data or fixing validation errors, use JSON patch operations via the apply_json_patches tool
3. JSON patches allow precise, targeted updates without losing correct data - when using this consider the extraction tools data model

JSON PATCH FORMAT:
Use RFC 6902 format for patches:
- {"op": "replace", "path": "/field_name", "value": "new_value"} - Update a field
- {"op": "add", "path": "/new_field", "value": "value"} - Add a field
- {"op": "remove", "path": "/field_name"} - Remove a field

When validation fails, generate targeted patches to fix only the problematic fields instead of regenerating the entire structure.

MULTI-IMAGE SUPPORT:
When you have access to multiple images (from PDFs with multiple pages or multiple image blocks), each image is numbered starting from 0:
- image_index=0: First image
- image_index=1: Second image
- image_index=2: Third image, etc.

Use the image_index parameter in image manipulation tools to specify which image to modify.

VALIDATION AND CORRECTION:
1. Review each field in your extracted data
2. Double-check each value against the source
3. Ensure you haven't mixed up similar-looking data
4. Pay special attention to small text and dates
5. When fixing errors, use JSON patches to target specific problems

CRITICAL DATE EXTRACTION RULES:
- Match each date value to its specific label/context, not just its visual prominence
- Small dates (like issue dates) are often less prominent but equally important
- When multiple dates appear close together, trace each one back to its specific label
- If a date field is unclear, use image enhancement tools before extraction
- If a data model allows None for a field, make sure to double check that there is no info for it before you set it to none.

Before finalizing extraction, mentally verify: "Does this date make logical sense for this specific field type?"
"""


async def structured_output_async(
    model: Any,  # AllowedModels
    data_format: Type[BaseModel],
    prompt: Union[str, Message, Image.Image],
    reprompt_cycles: int = 5,
    enable_image_tools: bool = True,
    existing_data: Optional[BaseModel] = None,
) -> Tuple[BaseModel, BedrockInvokeModelResponse]:
    """
    Extract structured data using Strands agents with tool-based validation.

    This recreates the structured_output_async functionality from ai-tools-registry
    using dynamically created tools that validate against the Pydantic model.

    Args:
        bedrock_client: AWS Bedrock client (not used directly with Strands)
        model: Model identifier
        data_format: Pydantic model class defining the expected structure
        prompt: Input content (text, image, or content blocks)
        reprompt_cycles: Maximum extraction attempts
        temperature: Model temperature (0-1)
        enable_image_tools: Whether to enable image enhancement tools
        existing_data: Optional existing data to update via patches

    Returns:
        Tuple of (extracted data, token usage dict)
    """
    # Create the dynamic extraction tool for this specific model
    extraction_tool, apply_json_patches = create_dynamic_extraction_tool_and_patch_tool(
        data_format
    )

    # Prepare tools list
    tools = [
        extraction_tool,  # The dynamically created tool
        apply_json_patches,
        load_image,
    ]

    if enable_image_tools:
        tools.append(enhance_image)

    # Get model ID from the model object
    if hasattr(model, "get_model_arn"):
        model_id = model.get_model_arn()
        logger.info(f"🤖 Using model ARN: {model_id}")
    else:
        model_id = str(model)
        logger.info(f"🤖 Using model as string: {model_id}")

    # Log the schema for debugging
    agent = Agent(
        model=model_id,
        tools=tools,
        system_prompt=f"{SYSTEM_PROMPT}\n\nExpected Schema:\n{json.dumps(data_format.model_json_schema(), indent=2)}",
        state={
            "current_extraction": None,
            "validation_result": None,
            "images": {},
            "existing_data": existing_data,
        },
    )
    logger.info("✅ Agent created successfully")

    # Process prompt based on type
    if isinstance(prompt, Image.Image):
        # Store image and convert to prompt format
        agent.state.set(key="images", value={0: prompt})

        # Convert PIL Image to bytes

        img_buffer = io.BytesIO()
        prompt.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        prompt_content = [
            Message(
                role="user",
                content=[
                    ContentBlock(text="Extract structured data from this image:"),
                    ContentBlock(
                        image=ImageContent(
                            format="png", source=ImageSource(bytes=img_bytes)
                        )
                    ),
                ],
            )
        ]
    elif isinstance(prompt, dict) and "content" in prompt:
        prompt_content = [prompt]
        # Extract and store any images from content blocks
        image_id = 0
        for block in prompt["content"]:
            if "image" in block:
                agent.state.set(
                    "images",
                    {
                        image_id: Image.open(
                            io.BytesIO(block["image"]["source"]["bytes"])
                        )
                    },
                )
                image_id += 1
    else:
        prompt_content = str(prompt)

    # Track token usage
    token_usage = {
        "inputTokens": 0,
        "outputTokens": 0,
        "totalTokens": 0,
        "cacheReadInputTokens": 0,
        "cacheWriteInputTokens": 0,
    }

    # Main extraction loop
    result = None
    response = None
    for cycle in range(reprompt_cycles):
        logger.debug(
            f"🔄 Cycle {cycle + 1}/{reprompt_cycles}: Starting structured output attempt"
        )

        # Prepare prompt for this cycle
        if existing_data:
            current_prompt = f"Please update the existing data using the extraction tool or patches. Existing data: {existing_data.model_dump_json()}\n\n{prompt_content}"
        else:
            current_prompt = prompt_content

        # Invoke agent
        response = await agent.invoke_async(current_prompt)
        logger.info(
            f"✅ Agent response received for cycle {cycle + 1}",
            extra={"prompt_cycle": cycle + 1, "agent_message": response},
        )
        token_usage["cacheReadInputTokens"] += response.metrics.accumulated_usage.get(
            "cacheReadInputTokens", 0
        )
        token_usage["cacheWriteInputTokens"] += response.metrics.accumulated_usage.get(
            "cacheWriteInputTokens", 0
        )
        token_usage["inputTokens"] += response.metrics.accumulated_usage.get(
            "inputTokens", 0
        )
        token_usage["outputTokens"] += response.metrics.accumulated_usage.get(
            "outputTokens", 0
        )
        token_usage["totalTokens"] += response.metrics.accumulated_usage.get(
            "totalTokens", 0
        )

        # Check for extraction in state
        current_extraction = agent.state.get("current_extraction")
        # If we got a successful extraction from the dynamic tool
        result = None
        if current_extraction:
            if isinstance(current_extraction, dict):
                logger.info(f"🔄 Converting dict extraction to {data_format.__name__}")
                try:
                    result = data_format(**current_extraction)
                    break
                except Exception as e:
                    logger.info(f"extraction is not valid, {e}")

                logger.info(f"✅ Successfully created {data_format.__name__} instance")

    # Return best effort result
    if result and response:
        return result, BedrockInvokeModelResponse(
            response=BedrockResponse(
                output=BedrockOutput(
                    message=BedrockMessage(
                        role="assistant",
                        content=[BedrockMessageContent(text=str(response))],
                    )
                )
            ),
            metering={model_id: BedrockUsage(**token_usage)},
        )

    logger.error(f"❌ Failed to extract structured data after {reprompt_cycles} cycles")
    raise ValueError("Failed to generate valid structured output.")


def structured_output(
    model: Any,  # AllowedModels
    data_format: Type[BaseModel],
    prompt: Union[str, Message, Image.Image],
    reprompt_cycles: int = 5,
    enable_image_tools: bool = True,
    existing_data: Optional[BaseModel] = None,
) -> Tuple[BaseModel, BedrockInvokeModelResponse]:
    """
    Synchronous version of structured_output_async.

    Extract structured data using Strands agents with tool-based validation.
    This is a wrapper that runs the async version in a sync event loop.

    Args:
        model: Model identifier
        data_format: Pydantic model class defining the expected structure
        prompt: Input content (text, image, or content blocks)
        reprompt_cycles: Maximum extraction attempts
        enable_image_tools: Whether to enable image enhancement tools
        existing_data: Optional existing data to update via patches

    Returns:
        Tuple of (extracted data, token usage dict)
    """
    return asyncio.run(
        structured_output_async(
            model=model,
            data_format=data_format,
            prompt=prompt,
            reprompt_cycles=reprompt_cycles,
            enable_image_tools=enable_image_tools,
            existing_data=existing_data,
        )
    )


if __name__ == "__main__":

    class Persona(BaseModel):
        age: int
        name: str

    async def async_main():
        result = await structured_output_async(
            model="us.anthropic.claude-sonnet-4-20250514-v1:0",
            data_format=Persona,
            prompt="Peter is a random person that works at AWS and is 23 years old.",
        )
        print(result)

    asyncio.run(async_main())
