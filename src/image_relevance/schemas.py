import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Metadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, str_strip_whitespace=True, revalidate_instances="always"
    )

    subject: str = Field(min_length=1, max_length=100)
    category: str = Field(min_length=1, max_length=50)
    attributes: list[str] = Field(max_length=10)
    caption: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)

    @field_validator("attributes")
    @classmethod
    def clean_attributes(cls, values: list[str]) -> list[str]:
        cleaned = []
        for value in values:
            value = value.strip()
            if not value or len(value) > 100:
                raise ValueError("Attributes must contain 1 to 100 characters")
            cleaned.append(value)
        return cleaned

class PostInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=200, examples=["The behavior of red foxes"])
    content: str = Field(
        min_length=1, max_length=6000,
        examples=["Red foxes hunt small mammals and adapt to many habitats."],
    )


class BatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_ids: list[str] = Field(default_factory=list, max_length=200)
    post_ids: list[str] = Field(default_factory=list, max_length=200)

    @field_validator("image_ids", "post_ids")
    @classmethod
    def valid_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("Resource IDs must be unique")
        for value in values:
            if len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid resource ID")
        return values


def valid_vector(values: list[float]) -> list[float]:
    if len(values) != 384 or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        raise ValueError("Expected 384 finite embedding values")
    norm = math.sqrt(sum(value * value for value in values))
    if not 0.99 < norm < 1.01:
        raise ValueError("Embedding must be normalized")
    return values
