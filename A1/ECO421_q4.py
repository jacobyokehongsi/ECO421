from selenium import webdriver # We'll use selenium web driver
driver = webdriver.Chrome('chromedriver')
driver.set_page_load_timeout(10)
driver.get('https://www.google.com/search?q=Machine+Learning')

# first_link = driver.find_element_by_tag_name('h3')
# first_link.click()

link = driver.find_element_by_xpath('//*[@class="LC20lb DKV0Md"]')
link.click()

# Close the window
driver.close()
