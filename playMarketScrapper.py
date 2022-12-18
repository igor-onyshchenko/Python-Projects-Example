

import requests  
from lxml import html


class playMarketScrapper:
    
    def __init__(self):
        pass
    
    def parse_body(self, url):
        self.url = url
        response = requests.get(self.url)

        # Parse the body into a tree
        parsed_body = html.fromstring(response.content)

        # Perform xpaths on the tree
        # Application name
        self.app_name = parsed_body.xpath('//div[@class="document-title"]/div/text()')[0]


        # Application developer name
        self.dev_name = parsed_body.xpath('//a[@class="document-subtitle primary"]/span/text()')[0]
        
        # Application category
        self.app_category = parsed_body.xpath('//a[@class="document-subtitle category"]/span[@itemprop="genre"]/text()')[0]
        
        #Application top developer  
        choice = parsed_body.xpath('//span[@class="badge-title"]/text()')
        
        if "Editors' Choice" in choice:
            self.editor_choice = 1
        else:
            self.editor_choice = 0
            
        if "Top Developer" in choice:
            self.top_dev = 1
        else:
            self.top_dev = 0
        
        
        #Application short description
        self.short_dscr = parsed_body.xpath('//div[@class="id-app-orig-desc"]/text()')[0]

        #Application full description
        dscr = parsed_body.xpath('//div[@class="id-app-orig-desc"]/p/text()')
        self.full_dscr = ''
        for i in range(len(dscr)):
            self.full_dscr = self.full_dscr + dscr[i]
            
        #Screenshots
        self.screenshots = parsed_body.xpath('//div[@class="thumbnails"]/img/@src')
        self.n_screenshots = len(screenshots)
        
       
        # Application score 
        self.app_score = float(parsed_body.xpath('//div[@class="score"]/text()')[0])
        
        # Application rating 
        self.app_rating = int((parsed_body.xpath('//div[@class="reviews-stats"]/span/text()')[0]).replace(',', ''))
        
        # Number of 5-Stars votes 
        self.votes_5 = int((parsed_body.xpath('//div[@class="rating-bar-container five"]/span[3]/text()')[0]).replace(',', ''))
        
        # Number of 4-Stars votes  
        self.votes_4 = int((parsed_body.xpath('//div[@class="rating-bar-container four"]/span[3]/text()')[0]).replace(',', ''))
        
        # Number of 3-Stars votes  
        self.votes_3 = int((parsed_body.xpath('//div[@class="rating-bar-container three"]/span[3]/text()')[0]).replace(',', ''))
        
        # Number of 2-Stars votes  
        self.votes_2 = int((parsed_body.xpath('//div[@class="rating-bar-container two"]/span[3]/text()')[0]).replace(',', ''))
        
        # Number of 1-Stars votes
        self.votes_1 = int((parsed_body.xpath('//div[@class="rating-bar-container one"]/span[3]/text()')[0]).replace(',', ''))
        
        
        #Additional information -- published/updated date 
        publish_date = parsed_body.xpath('//div[@itemprop="datePublished"]/text()')[0].replace(',', '').split()
        self.publish_month, self.publish_day, self.publish_year = publish_date
        
        #Additional information -- file size 
        self.file_size = parsed_body.xpath('//div[@itemprop="fileSize"]/text()')[0].strip()
        
        #Additional information -- number of downloads
        self.n_downloads = parsed_body.xpath('//div[@itemprop="numDownloads"]/text()')[0].strip()
        
        #Additional information -- application version 
        self.app_version = parsed_body.xpath('//div[@itemprop="softwareVersion"]/text()')[0].strip()
        
        #Additional information -- operating system Android requirement
        self.android_version_req = parsed_body.xpath('//div[@itemprop="operatingSystems"]/text()')[0].strip()
        
        #Additional information -- content rating
        self.content_rating = parsed_body.xpath('//div[@itemprop="contentRating"]/text()')[0]
        
        
        # Main Comments -- User-0
        try:
            self.user0_comment_head = parsed_body.xpath('//div[@data-expand-to="user-0"]/div[2]/div[2]/div/span/text()')[0]
        except:
            self.user0_comment_head = ''
        self.user0_comment_body = parsed_body.xpath('//div[@data-expand-to="user-0"]/div[2]/div[2]/div/text()')[1].strip()
        
        # Main Comments -- User-1 
        try:
            self.user1_comment_head = parsed_body.xpath('//div[@data-expand-to="user-1"]/div[2]/div[2]/div/span/text()')[0]
        except:
            self.user1_comment_head = ''
        self.user1_comment_body = parsed_body.xpath('//div[@data-expand-to="user-1"]/div[2]/div[2]/div/text()')[1].strip()
        
        # Main Comments -- User-2
        try:
            self.user2_comment_head = parsed_body.xpath('//div[@data-expand-to="user-2"]/div[2]/div[2]/div/span/text()')[0]
        except:
            self.user2_comment_head = ''
        self.user2_comment_body = parsed_body.xpath('//div[@data-expand-to="user-2"]/div[2]/div[2]/div/text()')[1].strip()
        
        # Main Comments -- User-3
        try:
            self.user3_comment_head = parsed_body.xpath('//div[@data-expand-to="user-3"]/div[2]/div[2]/div/span/text()')[0]
        except:
            self.user3_comment_head = ''
        self.user3_comment_body = parsed_body.xpath('//div[@data-expand-to="user-3"]/div[2]/div[2]/div/text()')[1].strip()
        



    def export_to_file(self, file_name, file_type, separation):
        
        file_path = file_name+"."+file_type
        self.wr = open(file_path,'w',encoding="utf8")
        self.separation = separation
        print('app_name', 'dev_name', 'app_category', 'editor_choice', 'top_dev', 'app_score',\
        'app_rating', 'votes_5', 'votes_4', 'votes_3', 'votes_2', 'votes_1', 'publish_day',\
        'publish_month', 'publish_year', 'file_size', 'n_downloads', 'app_version',\
        'android_version_req', 'content_rating', 'user0_comment_head', 'user0_comment_body',\
        'user1_comment_head', 'user1_comment_body', 'user2_comment_head', 'user2_comment_body',\
        'user3_comment_head', 'user3_comment_body', 'short_dscr', 'full_dscr', 'screenshots_links',\
        'n_screenshots', sep = self.separation, file = self.wr)
    
    def print_parsed(self):    
        print(self.app_name, self.dev_name, self.app_category, self.editor_choice, self.top_dev, self.app_score,\
        self.app_rating, self.votes_5, self.votes_4, self.votes_3, self.votes_2, self.votes_1, self.publish_day,\
        self.publish_month, self.publish_year, self.file_size, self.n_downloads, self.app_version,\
        self.android_version_req, self.content_rating, self.user0_comment_head, self.user0_comment_body,\
        self.user1_comment_head, self.user1_comment_body, self.user2_comment_head, self.user2_comment_body,\
        self.user3_comment_head, self.user3_comment_body, self.short_dscr, self.full_dscr, \
        self.screenshots, self.n_screenshots, sep = self.separation, file = self.wr)

		

##Example of class in work

scrapper = playMarketScrapper()

urls = ['https://play.google.com/store/apps/details?id=com.preloaded.RuggedRovers&hl=en',\
        'https://play.google.com/store/apps/details?id=com.viber.voip&hl=en',\
        'https://play.google.com/store/apps/details?id=com.mediocre.commute&hl=en']

scrapper.export_to_file("d:/play_market_log", "csv", ";")
for i in urls:
    scrapper.parse_body(i)
    scrapper.print_parsed()
    
scrapper.wr.close()    
