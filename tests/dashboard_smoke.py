"""Run inside the dashboard container with the development environment up."""
from streamlit.testing.v1 import AppTest

at=AppTest.from_file('app.py',default_timeout=40).run()
assert not at.exception, [e.message for e in at.exception]
assert not at.error, [e.value for e in at.error]
print('PASS operations page')
at.switch_page('pages/1_Player_Analytics.py').run()
assert not at.exception, [e.message for e in at.exception]
assert not at.error, [e.value for e in at.error]
print('PASS player analytics page and SQL queries')
