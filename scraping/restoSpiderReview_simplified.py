# Logging packages
import logging
import logzero
from logzero import logger
from scrapy.selector import Selector
import requests


# Scrapy packages
import scrapy
from TA_scrapy.items import ReviewRestoItem     # you can use it if you want but it is not mandatory
from TA_scrapy.spiders import get_info          # package where you can write your own functions


class RestoReviewSpider(scrapy.Spider):
    name = "RestoReviewSpider"

    def __init__(self, *args, **kwargs): 
        super(RestoReviewSpider, self).__init__(*args, **kwargs)

        # Set logging level
        logzero.loglevel(logging.WARNING)

        # To track the evolution of scrapping
        self.main_nb = 0
        self.resto_nb = 0
        self.review_nb = 0

        

    def start_requests(self):
        """ Give the urls to follow to scrapy
        - function automatically called when using "scrapy crawl my_spider"
        """

        # Basic restaurant page on TripAdvisor GreaterLondon
        url = 'https://www.tripadvisor.co.uk/Restaurants-g187147-Paris_Ile_de_France.html'
        yield scrapy.Request(url=url, callback=self.parse)

   
    def parse(self, response):
        """MAIN PARSING : Start from a classical reastaurant page
            - Usually there are 30 restaurants per page
            - 
        """

        # Display a message in the console
        logger.warn(' > PARSING NEW MAIN PAGE OF RESTO ({})'.format(self.main_nb))
        self.main_nb += 1

        # Get the list of the 30 restaurants of the page
        restaurant_urls = get_info.get_urls_resto_in_main_search_page(response)
        #logger.warn(' > RESTAURANT URLS ({})'.format(restaurant_urls))
        
        # For each url : follow restaurant url to get the reviews
        for restaurant_url in restaurant_urls:
            restaurant_url = 'https://www.tripadvisor.co.uk' + str(restaurant_url)
            yield response.follow(url=restaurant_url, callback=self.parse_resto)

        
        # Get next page information
        # Have we reached the article limit ?
        go_on = self.main_nb < 4

        # Deal with next page
        css_locator = 'a.nav.next.rndBtn.ui_button.primary.taLnk ::attr(href)'
        next_page = response.css(css_locator).extract_first()
        if next_page is not None and go_on :
            yield response.follow(url = 'https://www.tripadvisor.co.uk' + next_page, callback=self.parse)


    def parse_resto(self, response):
        """SECOND PARSING : Given a restaurant, get each review url and get to parse it
            - Usually there are 10 comments per page
        """
        #logger.warn(' > PARSING NEW RESTO PAGE ({})'.format(self.resto_nb))
        logger.warn(' Response ({})'.format(response))
      
        xpath_locator = '//div[@class="quote"]/a/@href'
        urls_review = response.xpath(xpath_locator).getall()
        ########################
        ########################

        # For each review open the link and parse it into the parse_review method
        for url_review in urls_review:
            yield response.follow(url=url_review, callback=self.parse_review)
        
       

        
        go_on2 = self.review_nb < 30000
        css_locator2 = 'a.nav.next.ui_button.primary ::attr(href)'
        next_page2 = response.css(css_locator2).extract_first()
        if next_page2 is not None and go_on2:
            yield response.follow(next_page2, callback=self.parse_resto)


    def parse_review(self, response):
        """FINAL PARSING : Open a specific page with review and client opinion
            - Read these data and store them
            - Get all the data you can find and that you believe interesting
        """
        # Count the number of review scrapped
        self.review_nb += 1

        logger.warn(' > PARSING NEW REVIEW NUMBER ({})'.format(self.review_nb))
        logger.warn(' Response ({})'.format(response))
        ########################
        #### YOUR CODE HERE ####
        ########################

        # You can store the scrapped data into a dictionnary or create an Item in items.py (cf XActuItem and scrapy documentation)

        review_item = {}

        xpath_locator_review = '//*[@class="partial_entry"]/text()'
        xpath_locator_date = '//*[@class="ratingDate relativeDate"]/@title' 
        xpath_locator_name = '//*[@id="listing_main_sur"]/div[1]/div[2]/div/a/text()'
        xpath_locator_grade = '//div[@class="rating"]/span/@alt' 
        xpath_locator_username = '//div[@class="username mo"]/span/text()'
        xpath_locator_adresse = '//*[@id="listing_main_sur"]/div[1]/div[2]/div/div[1]/address/span/span/span[1]/text()'
        xpath_locator_price = '//*[@id="listing_main_sur"]/div[1]/div[3]/div[3]/div[1]/text()'
        xpath_locator_style = '//*[@id="listing_main_sur"]/div[1]/div[3]/div[3]/div[2]/a/text()'
        xpath_locator_rank = '//*[@id="listing_main_sur"]/div[1]/div[3]/div[1]/div/span/div/b/span/text()'
        xpath_locator_tittle = '//*[@id="PAGEHEADING"]/text()'

        rep_review = response.xpath(xpath_locator_review).get()  
        rep_date = response.xpath(xpath_locator_date).get()
        rep_name = response.xpath(xpath_locator_name).get()
        rep_grade = response.xpath(xpath_locator_grade).get()
        rep_username = response.xpath(xpath_locator_username).get()
        rep_adress = response.xpath(xpath_locator_adresse).get()
        rep_price = response.xpath(xpath_locator_price).extract()
        rep_style = response.xpath(xpath_locator_style).extract()
        rep_rank = response.xpath(xpath_locator_rank).get()
        rep_tittle = response.xpath(xpath_locator_tittle).extract_first()
        
        
        review_item["Restaurant"] = rep_name
        review_item["Adresse"] = rep_adress
        review_item["Rank"] = rep_rank
        review_item["Date"] = rep_date
        review_item["Tittle"] = rep_tittle
        review_item["Review"] = rep_review
        review_item["Grade"] = rep_grade[0]
        review_item["Username"] = rep_username
        review_item["Price"] = rep_price[1]
        review_item["Style"] = rep_style
        

        
        logger.warn(' rep({})'.format(review_item))
        
        yield review_item


