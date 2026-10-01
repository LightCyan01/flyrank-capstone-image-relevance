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

class BatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_ids: list[str] = Field(min_length=1, max_length=200)

    @field_validator("image_ids")
    @classmethod
    def valid_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("Image IDs must be unique")
        for value in values:
            if len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Invalid image ID")
        return values
