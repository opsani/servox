# TODO: will be moved to seperate repo
def pytest_configure(config):
    config.addinivalue_line("markers", "incoming_webhook: marks slack notification tests as compatible with incoming webhook configuration")
    config.addinivalue_line("markers", "web_api: marks slack notification tests as compatible with web api configuration")