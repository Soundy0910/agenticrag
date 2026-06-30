"""
backend/graph_rag/extract.py

LLM-based entity and relationship extraction from parsed text.

Flow:
  text + GraphSchema → LLM prompt → structured JSON → ExtractionResult
  ExtractionResult → upsert_to_neo4j() → Neo4j graph

DESIGN CHOICES:
  - One LLM call per document (not per sentence): fewer API calls, the LLM
    can see the full context to resolve pronouns ("they" = the company named
    two sentences earlier).
  - JSON schema is derived from GraphSchema.to_prompt_block() so adding a new
    entity type to the schema automatically updates the extraction prompt.
  - Upsert strategy: MERGE on natural keys (Company.name, FiscalYear.year,
    Segment.name+company) so re-ingesting the same doc is idempotent.
  - Metrics use CREATE (not MERGE) because the same metric name ("Revenue")
    appears multiple times with different values across companies/years.
    The entity ID from the extraction JSON is used as a unique Neo4j property
    to allow idempotent re-runs via MERGE on that id.
"""

import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

import backend.config as cfg
from backend.graph_rag.schema import GraphSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntity:
    id: str           # local ID scoped to this extraction (e.g. "apple_co")
    type: str         # EntityType.name from the schema (e.g. "Company")
    properties: dict  # property key/values matching EntityType.properties


@dataclass
class ExtractedRelationship:
    type: str         # RelationshipType.name (e.g. "REPORTED")
    from_id: str      # ExtractedEntity.id of the source node
    to_id: str        # ExtractedEntity.id of the target node
    properties: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    doc_id: str
    collection: str
    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

_openai: OpenAI | None = None


def _llm() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=cfg.OPENAI_API_KEY)
    return _openai


_EXTRACTION_SYSTEM = """\
You are a structured data extractor. Given a financial text and a graph schema,
extract all entities and relationships present in the text.

Rules:
- Only extract what is explicitly stated — do not infer or hallucinate.
- Each entity needs a short unique id (snake_case, scoped to this response).
- Metric values must be numeric (float). Strip currency symbols and convert
  units to the most natural form (e.g. "$394.3 billion" → value=394.3, unit="billion USD").
- Omit any entity or relationship not supported by the provided schema.
- Return ONLY valid JSON matching the schema below. No explanation, no markdown.

Required JSON structure:
{
  "entities": [
    {"id": "<local_id>", "type": "<EntityType>", "properties": {<key>: <value>, ...}}
  ],
  "relationships": [
    {"type": "<RelationshipType>", "from_id": "<local_id>", "to_id": "<local_id>", "properties": {}}
  ]
}
"""


def extract_graph(
    text: str,
    schema: GraphSchema,
    doc_id: str,
    collection: str,
) -> ExtractionResult:
    """
    Extract entities and relationships from text using the provided schema.

    Parameters
    ----------
    text : str
        Source text (one parsed document or passage).
    schema : GraphSchema
        Defines which entity/relationship types to look for.
    doc_id : str
        Origin document identifier — stored on relationships as source_doc.
    collection : str
        Pinecone/Neo4j namespace this document belongs to.

    Returns
    -------
    ExtractionResult
        Structured entities and relationships ready for Neo4j upsert.
        Empty lists if the LLM returns no results or if text has no
        relevant entities.
    """
    user_prompt = (
        f"{schema.to_prompt_block()}\n\n"
        f"Text to extract from:\n\"\"\"\n{text[:6000]}\n\"\"\"\n\n"
        f"Extract entities and relationships. Return JSON only."
    )

    try:
        resp = _llm().chat.completions.create(
            model=cfg.DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=2000,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
    except Exception as exc:
        logger.error("extract_graph: LLM/parse error for doc %r: %s", doc_id, exc)
        return ExtractionResult(doc_id=doc_id, collection=collection, entities=[], relationships=[])

    entities = [
        ExtractedEntity(
            id=e["id"],
            type=e["type"],
            properties=e.get("properties", {}),
        )
        for e in data.get("entities", [])
        if e.get("type") in schema.entity_names()
    ]

    id_set = {e.id for e in entities}
    relationships = [
        ExtractedRelationship(
            type=r["type"],
            from_id=r["from_id"],
            to_id=r["to_id"],
            properties={**r.get("properties", {}), "source_doc": doc_id},
        )
        for r in data.get("relationships", [])
        if r.get("type") in schema.relationship_names()
        and r.get("from_id") in id_set
        and r.get("to_id") in id_set
    ]

    logger.info(
        "extract_graph: doc=%r entities=%d relationships=%d",
        doc_id, len(entities), len(relationships),
    )
    return ExtractionResult(
        doc_id=doc_id,
        collection=collection,
        entities=entities,
        relationships=relationships,
    )


# ---------------------------------------------------------------------------
# Neo4j upsert
# ---------------------------------------------------------------------------

def upsert_to_neo4j(result: ExtractionResult, driver) -> None:
    """
    Write an ExtractionResult into Neo4j using MERGE for idempotency.

    Merge keys by entity type:
      Company    → name
      FiscalYear → year
      Segment    → name (uniqueness scoped to company via relationship, not key)
      Metric     → extraction_id (unique per doc/extraction run)

    All properties are SET after MERGE so re-running updates stale values.
    """
    # Index entities by local id for relationship wiring.
    by_id: dict[str, ExtractedEntity] = {e.id: e for e in result.entities}

    with driver.session(database=cfg.NEO4J_DATABASE) as session:
        for e in result.entities:
            _upsert_node(session, e, result.collection)

        for r in result.relationships:
            src = by_id.get(r.from_id)
            tgt = by_id.get(r.to_id)
            if src and tgt:
                _upsert_relationship(session, r, src, tgt)


def _merge_key(entity: ExtractedEntity) -> dict:
    """Return the property dict used as the MERGE identity key for this entity type."""
    p = entity.properties
    if entity.type == "Company":
        return {"name": p.get("name", entity.id)}
    if entity.type == "FiscalYear":
        key = {"year": str(p.get("year", ""))}
        if p.get("quarter"):
            key["quarter"] = p["quarter"]
        return key
    if entity.type == "Segment":
        return {"name": p.get("name", entity.id)}
    # Metric: use extraction_id to allow multiple metrics with the same name
    return {"extraction_id": entity.id}


def _upsert_node(session, entity: ExtractedEntity, collection: str) -> None:
    key = _merge_key(entity)
    # SET all properties after MERGE, plus collection tag.
    props = {**entity.properties, "collection": collection}
    cypher = (
        f"MERGE (n:{entity.type} {{{', '.join(f'`{k}`: ${k}' for k in key)}}})\n"
        f"SET n += $props"
    )
    params = {**key, "props": props}
    session.run(cypher, **params)


def _upsert_relationship(
    session,
    rel: ExtractedRelationship,
    src: ExtractedEntity,
    tgt: ExtractedEntity,
) -> None:
    src_key = _merge_key(src)
    tgt_key = _merge_key(tgt)
    src_match = ", ".join(f"`{k}`: $src_{k}" for k in src_key)
    tgt_match = ", ".join(f"`{k}`: $tgt_{k}" for k in tgt_key)
    cypher = (
        f"MATCH (a:{src.type} {{{src_match}}})\n"
        f"MATCH (b:{tgt.type} {{{tgt_match}}})\n"
        f"MERGE (a)-[r:{rel.type}]->(b)\n"
        f"SET r += $props"
    )
    params = (
        {f"src_{k}": v for k, v in src_key.items()}
        | {f"tgt_{k}": v for k, v in tgt_key.items()}
        | {"props": rel.properties}
    )
    session.run(cypher, **params)
