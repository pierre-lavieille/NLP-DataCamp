################################################################################################
################################################################################################
#                                       TA NAVIGATION 
################################################################################################
################################################################################################
from scrapy.selector import Selector
import requests

def get_urls_resto_in_main_search_page(response):
    xpath_locator = '//*[@id="component_2"]/div/div/span/div/div/div/div/span/a/@href'
    return response.xpath(xpath_locator).getall()


def get_urls_next_list_of_restos(response):
    xpath = '//*[@id="EATERY_LIST_CONTENTS"]/div[3]/div/a'
    next_page = response.xpath(xpath).css('::attr(href)').extract()[-1]
    next_page_number = response.xpath(xpath).css('::attr(data-page-number)').extract_first()
    return next_page, next_page_number


def go_to_next_page(next_page, next_page_number=None, max_page=10, printing=False):
    """ According to next_page, and number of pages to scrap, tells if we should go on or stop.
    returns a boolean value : True (you should follow taht url) / False (you should stop scrapping)

    - next_page (str)           : the url of the next_page
    - next_page_number (int)    : often extracte from next_page, it is the number of the next page on the website
    - max_page (int)            : number of page you want to scrap.
                                If set to None, will scrap until the end of the website (might be very long).
    - printing(bool)            : If set to true will display messages at each new page to explain what is happening (useful for debug purprose)

    """

    if next_page is None:
        if printing: print(' - There is no next_page')
    else:
        if printing: print(' - There is a next_page')
        if printing: print(' - Page url is : {}'.format(next_page))
        if max_page is None:
            if printing: print(' - There is no number of page restriction. Go on.')
            return True
        else:
            if printing: print(' - Max page number is : {}'.format(max_page))

            if next_page_number is None:
                if printing: print(' -  No next number page : STOP.')
            else:
                if printing: print(' - Next page number is {}'.format(next_page_number))
                if int(next_page_number) <= int(max_page):
                    if printing: print(' - It is smaller than limit. Go on.')
                    return True
                else:
                    if printing: print('LIMIT was reached. STOP.')
    return False

################################################################################################
################################################################################################
#                                       REVIEW INFORMATION 
################################################################################################
################################################################################################
