import pytest

from tensor_cast.performance_model.profiling_database.data_source import (
    DataSourcePerformanceModel,
    QueryResult,
    QuerySource,
)


def test_query_source_enum():
    assert QuerySource.MEASURED.name == "MEASURED"
    assert QuerySource.INTERPOLATED.name == "INTERPOLATED"
    assert QuerySource.EXTRAPOLATED.name == "EXTRAPOLATED"


def test_query_result_creation():
    r = QueryResult(latency_us=45.3, confidence=1.0, source=QuerySource.MEASURED)
    assert r.latency_us == 45.3
    assert r.confidence == 1.0
    assert r.source == QuerySource.MEASURED
    assert r.details == {}


def test_query_result_with_details():
    r = QueryResult(
        latency_us=100.0,
        confidence=0.8,
        source=QuerySource.INTERPOLATED,
        details={"kernel_type": "MatMulV2", "csv_row": 42},
    )
    assert r.details["kernel_type"] == "MatMulV2"


def test_data_source_is_abstract():
    with pytest.raises(TypeError):
        DataSourcePerformanceModel()


def test_data_source_subclass_must_implement_lookup():
    class BadSource(DataSourcePerformanceModel):
        pass

    with pytest.raises(TypeError):
        BadSource()


def test_data_source_store_raises_by_default():
    class ReadOnlySource(DataSourcePerformanceModel):
        def lookup(self, op_invoke_info):
            return None

    source = ReadOnlySource()
    with pytest.raises(NotImplementedError, match="read-only"):
        source.store(None, None)
