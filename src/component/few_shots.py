few_shots = [
    {'Question' : "How many shoes do we have left for Nike in 35 size and signature collection?",
     'SQLQuery' : "SELECT sum(stock_quantity) FROM shoes WHERE brand = 'Nike' AND collection = 'Signature' AND size = '35'",
     'SQLResult': "Result of the SQL query",
     'Answer' : "18"},
    {'Question': "How much is the total price of the inventory for all size 40 shoes?",
     'SQLQuery':"SELECT SUM(price*stock_quantity) FROM shoes WHERE size = '40'",
     'SQLResult': "Result of the SQL query",
     'Answer': "27705"},
    {'Question': "If we have to sell all the Onisuka Tiger shoes today with discounts applied. How much revenue  our store will generate (post discounts)?" ,
     'SQLQuery' : """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, shoes_id from shoes where brand = 'Onisuka Tiger'
group by shoes_id) a left join discounts on a.shoes_id = discounts.shoes_id
 """,
     'SQLResult': "Result of the SQL query",
     'Answer': "13002.6"} ,
     {'Question' : "If we have to sell all the Crocs shoes today. How much revenue our store will generate without discount?" ,
      'SQLQuery': "SELECT SUM(price * stock_quantity) FROM shoes WHERE brand = 'Crocs'",
      'SQLResult': "Result of the SQL query",
      'Answer' : "36780"},
    {'Question': "How many vintage Adidas shoes do I have?",
     'SQLQuery' : "SELECT sum(stock_quantity) FROM shoes WHERE brand = 'Adidas' AND collection = 'Vintage'",
     'SQLResult': "Result of the SQL query",
     'Answer' : "86"
     },
    {'Question': "how much sales amount will be generated if we sell all size 45 shoes today in Puma brand after discounts?",
     'SQLQuery' : """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, shoes_id from shoes where brand = 'Puma' and size="45"
group by shoes_id) a left join discounts on a.shoes_id = discounts.shoes_id
 """,
     'SQLResult': "Result of the SQL query",
     'Answer' : "4598"
    }
]