"""
backend/graph_rag/schema.py

Config-driven graph schema definition.

DESIGN: schema-pluggable
  extract.py and query.py consume a GraphSchema object — they never reference
  entity type names or relationship names as string literals. Swapping the
  schema requires only a new GraphSchema instance, not changes to extraction
  or query logic.

New schema (v2) adds Filing, RiskFactor, Topic, and Chunk nodes so every
graph fact links back to a source chunk for citation grounding.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Schema building blocks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityType:
    """One node label in the graph."""
    name: str
    properties: list[str]
    description: str


@dataclass(frozen=True)
class RelationshipType:
    """One directed edge type in the graph."""
    name: str
    from_entity: str
    to_entity: str
    properties: list[str] = field(default_factory=list)


@dataclass
class GraphSchema:
    """Complete graph schema: entity types + relationship types."""
    name: str
    entities: list[EntityType]
    relationships: list[RelationshipType]

    def entity(self, name: str) -> EntityType:
        for e in self.entities:
            if e.name == name:
                return e
        raise KeyError(f"Entity type {name!r} not in schema {self.name!r}")

    def relationship(self, name: str) -> RelationshipType:
        for r in self.relationships:
            if r.name == name:
                return r
        raise KeyError(f"Relationship type {name!r} not in schema {self.name!r}")

    def entity_names(self) -> list[str]:
        return [e.name for e in self.entities]

    def relationship_names(self) -> list[str]:
        return [r.name for r in self.relationships]

    def to_prompt_block(self) -> str:
        """Render schema as a concise description block for LLM prompts."""
        lines = [f"Schema: {self.name}", "", "Entities:"]
        for e in self.entities:
            props = ", ".join(e.properties)
            lines.append(f"  {e.name}({props})  — {e.description}")
        lines += ["", "Relationships:"]
        for r in self.relationships:
            props = f"  [{', '.join(r.properties)}]" if r.properties else ""
            lines.append(f"  ({r.from_entity})-[{r.name}{props}]->({r.to_entity})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Finance / SEC schema v2
# ---------------------------------------------------------------------------

FINANCE_SCHEMA = GraphSchema(
    name="finance",
    entities=[
        EntityType(
            name="Company",
            properties=["company_id", "name", "ticker"],
            description="A public company (e.g. name='Microsoft', ticker='MSFT')",
        ),
        EntityType(
            name="Filing",
            properties=["filing_id", "filing_type", "fiscal_year", "filing_date", "source_file", "collection"],
            description="An SEC filing (e.g. filing_type='10-K', fiscal_year=2024)",
        ),
        EntityType(
            name="BusinessSegment",
            properties=["segment_id", "name"],
            description="A business segment within a company (e.g. 'Intelligent Cloud', 'Services')",
        ),
        EntityType(
            name="Metric",
            properties=["metric_id", "name", "value", "unit", "fiscal_year", "period"],
            description=(
                "A financial metric with a numeric value "
                "(e.g. name='Revenue', value=105362, unit='millions USD', fiscal_year=2025)"
            ),
        ),
        EntityType(
            name="RiskFactor",
            properties=["risk_id", "title", "summary", "fiscal_year"],
            description="A risk factor disclosed in a filing (e.g. 'AI infrastructure scaling risk')",
        ),
        EntityType(
            name="Topic",
            properties=["topic_id", "name"],
            description="A risk or business topic (e.g. 'AI', 'Cybersecurity', 'Regulation')",
        ),
        EntityType(
            name="Chunk",
            properties=["chunk_id", "source_file", "collection", "section", "text_preview"],
            description="A source text chunk from Pinecone — provides citation grounding for graph facts",
        ),
    ],
    relationships=[
        RelationshipType("FILED",            "Company",         "Filing",          []),
        RelationshipType("HAS_SEGMENT",      "Filing",          "BusinessSegment", []),
        RelationshipType("REPORTED_METRIC",  "BusinessSegment", "Metric",          ["source_doc"]),
        RelationshipType("SUPPORTED_BY",     "Metric",          "Chunk",           []),
        RelationshipType("HAS_RISK_FACTOR",  "Filing",          "RiskFactor",      []),
        RelationshipType("RELATED_TO_TOPIC", "RiskFactor",      "Topic",           []),
        RelationshipType("SUPPORTED_BY",     "RiskFactor",      "Chunk",           []),
        RelationshipType("HAS_CHUNK",        "Filing",          "Chunk",           []),
    ],
)


# ---------------------------------------------------------------------------
# Schema registry — looked up by collection name
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: dict[str, GraphSchema] = {
    "finance":     FINANCE_SCHEMA,
    "sec_filings": FINANCE_SCHEMA,
    "sec-filings": FINANCE_SCHEMA,
}


def get_schema(collection: str) -> GraphSchema:
    """Return the schema for a collection, defaulting to FINANCE_SCHEMA."""
    return SCHEMA_REGISTRY.get(collection, FINANCE_SCHEMA)
