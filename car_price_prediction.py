import urllib.request

import numpy as np
from bs4 import BeautifulSoup
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

'''
__Project_By__ =    Yogesh Wadekar,
                    Lokesh Malpote,
                    Tushar Shendge,
                    Onkar Nakate
'''

percent_of_test_data = 25

url_offers = 'https://www.otomoto.pl/osobowe/volkswagen/passat/b6-2005-2010/?search%5Bfilter_float_price%3Afrom%5D=5000&search%5Bfilter_float_price%3Ato%5D=50000&search%5Bfilter_float_year%3Afrom%5D=2005&search%5Bfilter_float_year%3Ato%5D=2010&search%5Bfilter_enum_fuel_type%5D%5B0%5D=diesel&search%5Bfilter_enum_damaged%5D=0&search%5Bfilter_enum_rhd%5D=0&search%5Border%5D=created_at%3Adesc&search%5Bcountry%5D='

'''INPUT'''

#                                  year,Running Km,Engine_capacity(cc)
cars_to_predict_price = np.array([[2006, 190000, 1896],
                                  [2009, 300000, 1968],
                                  ])


def get_html(url):
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, "lxml")
    return soup


def count_pages(main_url):
    soup = get_html(main_url)
    try:
        pgno=soup.find_all('span', attrs={'class': 'page'})[-2].text
        print(pgno)
    except:
        return 1
    else:
        return int(1)


def collect_data(main_url, pages=3):
    Xy = np.empty((0, 4))
    for page in range(1, pages + 1):
        print('\r\tParsing page: ' + str(page), end='')
        url = main_url
        soup = get_html(url)
        articles = soup.find_all('article')

        for article in articles:
            # year
            year_tag = article.find("li", attrs={'data-code': 'year'})

            if (year_tag == None):
                continue
            year = int(year_tag.text)
            # mileage
            mileage_tag = article.find("li", attrs={'data-code': 'mileage'})
            if (mileage_tag == None):
                continue
            mileage = int(mileage_tag.span.text.replace(' ', '')[:-2])
            # capacity
            capacity_tag = article.find("li", attrs={'data-code': 'engine_capacity'})
            if (capacity_tag == None):
                continue
            capacity = int(capacity_tag.span.text.replace(' ', '')[:-3])
            # price
            price_tag = article.find("span", attrs={'class': 'offer-price__number'})
            if (price_tag == None):
                continue
            price = int(float(price_tag.span.text.replace(' ', '').replace(",",".")))
            price=price*19.55                                                   ## convert to indian currency
            if price<0:
                continue
            Xy = np.append(Xy, [[year, mileage, capacity, price]], axis=0)

    print('\n')
    return Xy


def split_data(Xy, percent_of_test_data=30):
    n = len(Xy)
    np.random.shuffle(Xy)

    n_train = round((100 - percent_of_test_data) / 100 * n)
    n_test = n - n_train

    [Xy_train, Xy_test, _] = np.split(Xy, [n_train, n_train + n_test])

    X_train = Xy_train[:, [0, 1, 2]]
    y_train = Xy_train[:, [3]]

    X_test = Xy_test[:, [0, 1, 2]]
    y_test = Xy_test[:, [3]]

    return X_train, y_train, X_test, y_test


def main():
    print("\nCollecting data from %s..." % url_offers[:30])
    pages = count_pages(url_offers)
    print("\tFound %d pages" % pages)

    Xy = collect_data(url_offers, pages)
    print("\tCollected %d samples:" % len(Xy))
    X_train, y_train, X_test, y_test = split_data(Xy, percent_of_test_data)
    print('\t\tTraining samples: %d' % len(X_train))
    print('\t\tTest samples: %d' % len(X_test))

    print('\nLearning...')
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    np.set_printoptions(formatter={'float_kind': '{:f}'.format})
    print('\tInterceptor: ', regr.intercept_)
    print('\tCoefficients: ', regr.coef_)

    print('\nTesting...')
    y_pred = regr.predict(X_test)
    print("\tMean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    print('\tVariance score: %.2f' % r2_score(y_test, y_pred))

    print('\nPredicting...')
    # prices = regr.predict(cars_to_predict_price)
    for car in cars_to_predict_price:
        price = regr.predict([car])
        print('\tThe best price for specified car (year %d, mileage %d, capacity %d) is %.2f PLN' % (car[0], car[1], car[2], price[0][0]))


if __name__ == '__main__':
    main()


