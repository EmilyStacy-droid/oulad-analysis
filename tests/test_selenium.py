import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = "http://localhost:5000"

@pytest.fixture
def driver():
    driver = webdriver.Chrome()
    yield driver
    driver.quit()

def test_prediction_flow(driver):
    driver.get(BASE_URL)

    form = driver.find_element(By.ID, "predictForm")
    driver.find_element(By.ID, "submit").click()

    validation_message = form.find_element(By.ID, "sum_click").get_attribute("validationMessage")

    assert validation_message != ""
