"""
backend/graph_rag/schema.py

Config-driven graph schema definition.

DESIGN: schema-pluggable
  extract.py and query.py consume a GraphSchema object — they never reference
  entity type names or relationship names as string literals. Swapping the
  schema (e.g. finance → biomedical) requires only a new GraphSchema instance,
  not changes to extraction or query logic.

  The only built-in schema is FINANCE_SCHEMA, proven on SEC 10-K language.
  Register additional schemas in SCHEMA_REGISTRY if new collections need them.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Schema building blocks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityType:
    """One node label in the graph."""
    name: str                     # Neo4j label, e.g. "Company"
    properties: list[str]         # expected property keys on this node
    description: str              # for LLM extraction prompts


@dataclass(frozen=True)
class RelationshipType:
    """One directed edge type in the graph."""
    name: str                     # Neo4j relationship type, e.g. "REPORTED"
    from_entity: str              # label of the source node
    to_entity: str                # label of the target node
    properties: list[str] = field(default_factory=list)


@dataclass
class GraphSchema:
    """
    Complete graph schema: entity types + relationship types.

    Consumed by:
      extract.py  — builds LLM prompt from entity/relationship descriptions
      query.py    — builds Cypher node labels and relationship names
    """
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
# Finance / SEC schema
# ---------------------------------------------------------------------------

FINANCE_SCHEMA = GraphSchema(
    name="finance",
    entities=[
        EntityType(
            name="Company",
            properties=["name", "ticker", "sector"],
            description="A public or private company (e.g. 'Apple Inc.', ticker 'AAPL')",
        ),
        EntityType(
            name="Metric",
            properties=["name", "value", "unit"],
            description=(
                "A financial metric with a numeric value "
                "(e.g. name='Revenue', value=394.3, unit='billion USD')"
            ),
        ),
        EntityType(
            name="FiscalYear",
            properties=["year", "quarter"],
            description=(
                "A fiscal year or quarter "
                "(e.g. year='2024', quarter='Q4' or quarter=null for full year)"
            ),
        ),
        EntityType(
            name="Segment",
            properties=["name", "description"],
            description="A business segment or product division within a company",
        ),
    ],
    relationships=[
        RelationshipType("REPORTED",         "Company",    "Metric",     ["source_doc"]),
        RelationshipType("IN_PERIOD",         "Metric",     "FiscalYear", []),
        RelationshipType("HAS_SEGMENT",       "Company",    "Segment",    []),
        RelationshipType("SEGMENT_REPORTED",  "Segment",    "Metric",     ["source_doc"]),
    ],
)


# ---------------------------------------------------------------------------
# Schema registry — looked up by collection name
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: dict[str, GraphSchema] = {
    "finance":     FINANCE_SCHEMA,
    "sec_filings": FINANCE_SCHEMA,
    # "biomedical": BIOMEDICAL_SCHEMA,  # add when needed
}


def get_schema(collection: str) -> GraphSchema:
    """Return the schema for a collection, defaulting to FINANCE_SCHEMA."""
    return SCHEMA_REGISTRY.get(collection, FINANCE_SCHEMA)
