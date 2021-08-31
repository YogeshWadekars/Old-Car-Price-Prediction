
percent_of_test_data = 33
main_url = 'https://www.otomoto.pl/osobowe/volkswagen/passat/b6-2005-2010/?search%5Bfilter_float_price%3Afrom%5D=5000&search%5Bfilter_float_price%3Ato%5D=50000&search%5Bfilter_float_year%3Afrom%5D=2005&search%5Bfilter_float_year%3Ato%5D=2010&search%5Bfilter_enum_fuel_type%5D%5B0%5D=diesel&search%5Bfilter_enum_damaged%5D=0&search%5Bfilter_enum_rhd%5D=0&search%5Border%5D=created_at%3Adesc&search%5Bcountry%5D='

# %%

import sys
sys.path.append("C:\\Users\Yogesh Wadekar\Desktop\Kaliaa\car_price_prediction_master")
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import car_price_prediction

##%matplotlib inline
##this doesn't work in python,insead of this use following statement:
'exec(%matplotlib inline)'


plt.rcParams['figure.figsize'] = 16,5

# %%

pages = car_price_prediction.count_pages(main_url)
print("Found %d pages" % pages)

# %%

Xy = car_price_prediction.collect_data(main_url,pages)

print("Collected %d samples" % len(Xy))

# %%
print("\nLast 10 samples:")
print("[year, Running(Km), Engine_capacity(CC), price(Rs)]")
np.set_printoptions(precision=0)
print(Xy[-10:])

# %%

plt.figure()
plt.hist(Xy[:,1],30)
plt.title("Running(Km) distribution")
plt.xlabel("Running[Km]")
plt.show()

# %%
plt.figure()
plt.hist(Xy[:,3],30)
plt.title("Price distribution")
plt.xlabel("Price [RS]")
plt.show()

# %%
plt.figure()
plt.title("Price vs year")
plt.xlabel("Price [RS]")
plt.ylabel("Year")
plt.plot(Xy[:,3],Xy[:,0],"x")
plt.show()

# %%
plt.figure()
plt.title("Running(Km) vs year")
plt.xlabel("Running[km]")
plt.ylabel("Year")
plt.plot(Xy[:,1],Xy[:,0],"x")
plt.show()


# %%

X_train, y_train, X_test, y_test = car_price_prediction.split_data(Xy, percent_of_test_data) 

print('Training samples: %d' % len(X_train))
print('Test samples: %d' % len(X_test))

# %%

regr = linear_model.LinearRegression(normalize=True)
regr.fit(X_train, y_train)

np.set_printoptions(formatter={'float_kind': '{:f}'.format})

# %%

y_pred = regr.predict(X_test)

# %%
plt.figure()
plt.plot(y_test[-50:], label="Real")
plt.plot(y_pred[-50:], label="Predicted")
plt.legend()
plt.title("Price: real vs predicted")
plt.ylabel("price [RS]")
plt.xticks(())
plt.show()


# %%
"""
Passat B6 1.9 TDI 2009 with 188 000 Running Km
"""
# %%

price_pred = regr.predict([[2009,180000, 1896]])
print('The best price for volkswagen Passat B6 1.9 TDI 2009 with 188000 Running Km is %.2f RS' % price_pred[0][0])

# %%
"""
### Passat B6 2.0 TDI 2006 with 288 000 Running Km
"""
# %%
price_pred = regr.predict([[2006,288000, 1968]])
print('The best price for volkswagen Passat B6 2.0 TDI 2006 with 288000 Running Km is  %.2f RS' % price_pred[0][0])
