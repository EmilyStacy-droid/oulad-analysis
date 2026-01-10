import os
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import urllib3
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

SELENIUM_REMOTE_URL = os.getenv(
    "SELENIUM_REMOTE_URL",
    "http://selenium:4444/wd/hub"
)

BASE_URL = "http://oulad-app:5000"

def wait_for_selenium(url, timeout=20):
    http = urllib3.PoolManager()
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = http.request('GET', url.replace('/wd/hub','/status'))
            if r.status == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("Selenium server not ready after {} seconds".format(timeout))

@pytest.fixture
def driver():
    wait_for_selenium(SELENIUM_REMOTE_URL)
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Remote(
        command_executor=SELENIUM_REMOTE_URL,
        options=options
    )
    yield driver
    driver.quit()

def test_prediction_flow(driver):
    driver.get(BASE_URL)

    submit_btn = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "submit"))
    )
    submit_btn.click()

    input_el = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "sum_click"))
    )
    validation_message = input_el.get_attribute("validationMessage")

    assert validation_message != ""
