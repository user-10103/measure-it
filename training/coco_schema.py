"""One place that decides which COCO category is the facet class.

THREE scripts each assumed they knew the schema instead of reading it, and each
assumption produced a different silent corruption:

  merge_datasets.py     writes a HARDCODED categories block while copying
                        category_id UNCHANGED. The GIS export declares a single
                        [{id:1, name:'roof facet'}] class, so its 14,068 facets
                        keep id 1 and the merged file relabels them
                        roof_polygon -> prep then deletes every one of them.

  oversample_dataset.py writes the SAME hardcoded block over whatever the input
                        declared. Run on prep's output -- single class at id 1
                        named "roof facet" -- it renames that class to
                        roof_polygon in place, touching no annotation. The COCO
                        category NAME becomes SAM3's text prompt, so training
                        would have learned the concept "roof_polygon" from
                        pictures of facets for 15 epochs, converged cleanly, and
                        then been prompted at inference with "roof facet" -- a
                        string it never saw. Every count in every log correct.

  prep_sam3_facets.py   falls back to "no facet-named class -> keep ALL classes",
                        which silently admits roof outlines as facets.

All three also COUNT facets as `category_id == 2`, so a correct single-class
dataset reports "facet annotations: 0" -- the kind of false alarm that teaches
people to ignore the check.

Resolve by NAME, never by id. Refuse to guess.
"""
from __future__ import annotations

FACET_NAMES = {"facet", "roof_facet", "roof facet", "roof face"}

STD_CATEGORIES = [{"id": 1, "name": "roof_polygon", "supercategory": "roof"},
                  {"id": 2, "name": "facet", "supercategory": "roof"}]


def resolve_facet_ids(categories, *, strict: bool = True) -> set[int]:
    """Category ids that denote a roof FACET, by name.

    strict=True (default) raises when no category is facet-named rather than
    falling back to "keep everything" -- that fallback is how outlines get
    admitted as facets.
    """
    ids = {c["id"] for c in categories
           if str(c.get("name", "")).lower() in FACET_NAMES}
    if not ids and strict:
        raise ValueError(
            f"no facet-named category in {[c.get('name') for c in categories]}; "
            f"expected one of {sorted(FACET_NAMES)}. Refusing to guess which "
            f"class is the facet class.")
    return ids


def count_facets(annotations, categories) -> int:
    """Facet annotation count, resolved by name. Never `category_id == 2`."""
    ids = resolve_facet_ids(categories, strict=False)
    if not ids:
        return 0
    return sum(1 for a in annotations if a.get("category_id") in ids)
