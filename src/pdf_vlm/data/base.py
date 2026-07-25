"""Common dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from pdf_vlm.schemas import DatasetBundle, DatasetDocument, QAExample, QAPair


class BaseDataset(ABC):
    """Abstract adapter: load -> DatasetBundle with documents/pages/qa_pairs."""

    name: str = "base"

    @abstractmethod
    def load(self) -> DatasetBundle:
        raise NotImplementedError

    def __iter__(self) -> Iterator[DatasetDocument]:
        return iter(self.load().documents)

    def iter_examples(self) -> Iterator[QAExample]:
        bundle = self.load()
        for doc, qa in bundle.iter_qa():
            yield QAExample.from_qa_pair(doc, qa)

    def to_examples(self) -> list[QAExample]:
        return list(self.iter_examples())


class UnifiedDataset(BaseDataset):
    """In-memory wrapper around an already-built DatasetBundle."""

    def __init__(self, bundle: DatasetBundle):
        self._bundle = bundle
        self.name = bundle.name

    def load(self) -> DatasetBundle:
        return self._bundle


def bundle_to_examples(bundle: DatasetBundle) -> list[QAExample]:
    return [QAExample.from_qa_pair(doc, qa) for doc, qa in bundle.iter_qa()]


def filter_bundle_by_question_types(
    bundle: DatasetBundle,
    types: set[str] | None,
) -> DatasetBundle:
    if not types:
        return bundle
    wanted = {t.lower() for t in types}
    docs: list[DatasetDocument] = []
    for doc in bundle.documents:
        qas = [qa for qa in doc.qa_pairs if qa.question_type.value.lower() in wanted]
        if qas:
            docs.append(doc.model_copy(update={"qa_pairs": qas}))
    return DatasetBundle(
        name=bundle.name,
        documents=docs,
        split=bundle.split,
        meta={**bundle.meta, "filtered_question_types": sorted(wanted)},
    )
