## My own module to import data from ECOS(ECOnomic statistics System)
module DataCollector

export get_ECOS
using DataFrames
using HTTP, JSON3
# Define a function to call dataset from openAPI
function get_ECOS(code,freq,begindate,enddate,args="/",df=DataFrame())
    my_id = "FLZETTVUFHQRTT86M103";
    url = "http://ecos.bok.or.kr/api/StatisticSearch/"*my_id*"/json/kr/1/3000/"*code*"/"*freq*"/"*begindate*"/"*enddate*"/"*args;
    response = String((HTTP.get(url).body))
    jsdata = JSON3.read(response)
    DF=  DataFrame(jsdata.StatisticSearch.row)
    df=DataFrame(date=DF.TIME,data=parse.(Float64,DF.DATA_VALUE),name=DF.STAT_NAME.*" : ".*DF.ITEM_NAME1)
    return df
end
end