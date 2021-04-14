# import pandas
import pandas as pd
# import matplotlib
import matplotlib.pyplot as plt
# import seaborn
import seaborn as sns
import matplotlib.pyplot as plt



data_url = 'http://bit.ly/2cLzoxH'
gapminder = pd.read_csv(data_url)
gapminder_2007 = gapminder[gapminder['year']==2007]
print(gapminder.head(10))
bplot = sns.boxplot(x="variable",y="value",
                 data=pd.melt(gapminder_2007[['lifeExp','gdpPercap']]),
                 width=0.5,
                 palette="colorblind")



plt.show()


