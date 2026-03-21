# Legacy chatbot imports — only loaded on demand to avoid missing-dep errors
# when using just scid_parser / wyckoff_analyzer / plotter.
def __getattr__(name):
    _legacy = {
        'get_stock_data': '.data_processing',
        'add_technical_indicators': '.data_processing',
        'format_data_for_chart': '.data_processing',
        'WyckoffModelHandler': '.model_handler',
    }
    if name in _legacy:
        import importlib
        mod = importlib.import_module(_legacy[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'get_stock_data',
    'add_technical_indicators',
    'format_data_for_chart',
    'WyckoffModelHandler',
]