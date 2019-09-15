#Extraction French region WINE Searcher
library(RSelenium)
library(XML)
library(rvest)
library(httr)


#open the firefox server
rD <- rsDriver(port = 3227L, browser = "firefox", version = "latest")
#assign server session to variable mybrowser
mybrowser <- rD$client
#navigate to sub-region 
mybrowser$navigate("https://www.winemag.com/?s=&search_type=reviews&drink_type=wine&varietal=Bordeaux-style%20Red%20Blend&price=20-5000")


