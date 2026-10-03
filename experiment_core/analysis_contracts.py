"""Renderer-neutral, versioned analysis and long-form observations protocol."""
from typing import Annotated, Literal
from pydantic import Field, model_validator
from .contracts import Contract, Identifier

class AnalysisSpec(Contract):
    schema_version: Literal["analysis-spec/v1"] = "analysis-spec/v1"
    run_ids: list[Identifier] = Field(min_length=1, max_length=100)
    kind: Literal["comparison","sensitivity","heatmap","timeseries","paired"] = "comparison"
    metrics: list[Identifier] = Field(default_factory=lambda:["operating_profit"], min_length=1, max_length=6)
    x_parameter: str | None = None
    y_parameter: str | None = None
    confidence: Annotated[float, Field(gt=0, lt=1)] = .95
    width: Literal["single","double"] = "double"
    columns: Annotated[int, Field(ge=1, le=3)] = 2
    dpi: Annotated[int, Field(ge=300, le=1200)] = 600
    formats: list[Literal["pdf","svg","png"]] = Field(default_factory=lambda:["pdf","svg","png"], min_length=1)
    renderer: Literal["matplotlib"] = "matplotlib"
    labels: dict[Identifier, Annotated[str, Field(min_length=1, max_length=120)]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_axes(self):
        if len(set(self.run_ids)) != len(self.run_ids) or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("run IDs and metric panels must be unique")
        if self.kind in {"sensitivity","heatmap"} and not self.x_parameter:
            raise ValueError("sensitivity requires a numeric parameter axis")
        if self.kind == "heatmap" and (not self.y_parameter or self.x_parameter == self.y_parameter):
            raise ValueError("heatmap requires two different parameter axes")
        return self
