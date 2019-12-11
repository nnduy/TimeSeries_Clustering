import glob, re
import numpy as np
import pandas as pd

import os
import folium
from folium.plugins import MarkerCluster
from folium.map import *
from folium import plugins
from folium.plugins import MeasureControl
from folium.plugins import FloatImage
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plotnine import *
from datetime import datetime
from mizani.breaks import date_breaks

theme_set(theme_linedraw()) # default theme

# We run this to suppress various deprecation warnings from plotnine - keeps our notebook cleaner
import warnings
warnings.filterwarnings('ignore')

class VisualizeInput:
    def __init__(self):
        pass

    # read in all csv file to data variable
    data = {
        'avd': pd.read_csv('input/air_visit_data.csv', parse_dates=['visit_date']),
        # DataFrame.dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False)[source]
        # Remove missing values.
        'asi' : pd.read_csv('input/air_store_info.csv').dropna(),
        'hsi' : pd.read_csv('input/hpg_store_info.csv'),
        'ar' : pd.read_csv('input/air_reserve.csv', parse_dates=['visit_datetime', 'reserve_datetime']),
        'hr' : pd.read_csv('input/hpg_reserve.csv', parse_dates=['visit_datetime', 'reserve_datetime']),
        'idr' : pd.read_csv('input/store_id_relation.csv'),
        'tes': pd.read_csv('input/sample_submission.csv'),
        'hol': pd.read_csv('input/date_info.csv').rename(columns={'calendar_date':'visit_date'})
        }

    # ============ Step 00_a: Visualize input data - Air Visit Data ==============
    root = os.getcwd()
    plot_folder = "plots"
    dir_plot = os.path.join(root, plot_folder)

    def save_plot(self, plot, fn, path, w, h):
        ggsave(plot=plot, filename=fn, path=path, width=w, height=h, dpi=300, units="in", device='png')

    # We start with the number of visits to the air restaurants. Here we plot the total number of visitors per day over the
    # full training time range together with the median visitors per day of the week and month of the year:
    def visualize_air_visit(self):
        # dfa = data['avd']
        # dfa = dfa.groupby('visit_date')['visitors'].agg(['sum','count'])
        df_avd_origin = self.data['avd']
        df_avd = (df_avd_origin.groupby('visit_date')
           .agg({'air_store_id':'count', 'visitors': 'sum'})
           .reset_index()
           .rename(columns={'air_store_id':'air_store_id Count', 'visitors':'All visitors'})
        )

        plot_all_air_visits = (ggplot(df_avd)
            + geom_line(aes(x='visit_date', y='All visitors', group = 1), color='blue')
            + scale_x_datetime(breaks=date_breaks('3 months'))
            + labs(y="All visitors", x="Date")
             )
        # ggsave(plot=plot_all_air_visits, filename="all_air_visits", path=self.dir_plot, width=10, height=4, dpi=300, units="in", device='png')
        self.save_plot(plot_all_air_visits, "plot_all_air_visits", self.dir_plot, 10, 4)

        # print(df_avd_origin)
        plot_all_air_visits_histogram = (ggplot(df_avd_origin, aes('visitors'))
            + geom_histogram(fill = "blue", bins = 30)
            + scale_x_log10()
             )
        self.save_plot(plot_all_air_visits_histogram, "plot_all_air_visits_histogram", self.dir_plot, 10, 4)

        week_df = df_avd_origin.groupby(df_avd_origin['visit_date'].dt.weekday_name).mean()
        week_df.reset_index(level=['visit_date'], inplace=True)
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        week_df['visit_date'] = pd.Categorical(week_df['visit_date'], categories=cats, ordered=True)
        week_df = week_df.sort_values('visit_date')
        # print(week_df)
        plot_all_air_visits_weekday = (ggplot(week_df, aes(x='visit_date', y='visitors', fill='visit_date'))
            + geom_col()
            + theme(legend_position = "none", axis_text_x  = element_text(angle=45, hjust=1, vjust=0.9))
            + labs(x = "Day of the week", y = "Median visitors")
             )
        self.save_plot(plot_all_air_visits_weekday, "plot_all_air_visits_weekday", self.dir_plot, 5, 4)
        month_df = df_avd_origin.groupby(df_avd_origin['visit_date'].dt.strftime('%b')).mean()
        month_df.reset_index(level=['visit_date'], inplace=True)
        month_cats = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_df['visit_date'] = pd.Categorical(month_df['visit_date'], categories=month_cats, ordered=True)
        month_df = month_df.sort_values('visit_date')
        # print(month_df)
        plot_all_air_visits_months = (ggplot(month_df, aes(x='visit_date', y='visitors', fill='visit_date'))
            + geom_col()
            + theme(legend_position = "none")
            + labs(x = "Month", y = "Median visitors")
             )
        self.save_plot(plot_all_air_visits_months, "plot_all_air_visits_months", self.dir_plot, 5, 4)



    # ============ Step 00_b: Visualize input data - Air Reservations ==============
    def visualize_air_reserve(self):
        df_ar = self.data['ar']
        df_ar['reserve_date'] = df_ar['reserve_datetime'].dt.date
        df_ar['reserve_hour'] = df_ar['reserve_datetime'].dt.hour
        df_ar['reserve_wday'] = df_ar['reserve_datetime'].dt.dayofweek
        df_ar['visit_date'] = df_ar['visit_datetime'].dt.date
        df_ar['visit_hour'] = df_ar['visit_datetime'].dt.hour
        df_ar['visit_wday'] = df_ar['visit_datetime'].dt.dayofweek
        df_ar['diff_day'] = df_ar['visit_datetime'].sub(df_ar['reserve_datetime'], axis=0).dt.days
        df_ar['diff_hour'] = (df_ar['visit_datetime'].sub(df_ar['reserve_datetime'], axis=0)).astype('timedelta64[h]')
        print(df_ar)

        df_ar_all_visits_reserve = (df_ar.groupby('visit_date')
           .agg({'air_store_id':'count', 'reserve_visitors': 'sum'})
           .reset_index()
           .rename(columns={'air_store_id':'air_store_id_count', 'reserve_visitors':'all_visitors'})
        )
        print("df_ar_all_visits_reserve", df_ar_all_visits_reserve)
        plot_all_air_visits_reserve = (ggplot(df_ar_all_visits_reserve)
            + geom_line(aes(x='visit_date', y='all_visitors', group = 1), color='blue')
            + scale_x_datetime(breaks=date_breaks('3 months'))
            + labs(y="all_visitors", x="Date")
             )
        self.save_plot(plot_all_air_visits_reserve, "plot_all_air_visits_reserve", self.dir_plot, 10, 4)

        df_ar_all_visits_reserve_hour = (df_ar.groupby('visit_hour')
           .agg({'air_store_id':'count', 'reserve_visitors': 'sum'})
           .reset_index()
           .rename(columns={'air_store_id':'air_store_id_count', 'reserve_visitors':'all_visitors'})
        )
        plot_all_air_visits_reserve_hour = (ggplot(df_ar_all_visits_reserve_hour, aes(x='visit_hour', y='all_visitors'))
            + geom_col(fill = "blue")
             )
        self.save_plot(plot_all_air_visits_reserve_hour, "plot_all_air_visits_reserve_hour", self.dir_plot, 10, 4)

        df_ar_all_visits_reserve_diff = (df_ar.groupby('diff_hour')
           .agg({'air_store_id':'count', 'reserve_visitors': 'sum'})
           .reset_index()
           .rename(columns={'air_store_id':'air_store_id_count', 'reserve_visitors':'all_visitors'})
        )
        df_ar_all_visits_reserve_diff = df_ar_all_visits_reserve_diff.loc[df_ar_all_visits_reserve_diff['diff_hour'] < 24*5]
        print(df_ar_all_visits_reserve_diff)
        plot_all_air_visits_reserve_diff = (ggplot(df_ar_all_visits_reserve_diff, aes(x='diff_hour', y='all_visitors'))
            + geom_col(fill = "blue")
            + labs(x = "Time from reservation to visit [hours]")
             )
        self.save_plot(plot_all_air_visits_reserve_diff, "plot_all_air_visits_reserve_diff", self.dir_plot, 10, 4)


    # ============ Step 00_c: Visualize input data - HPG Reservations ==============
    def visualize_hpg_reserve(self):
        df_hr = self.data['hr']
        df_hr['reserve_date'] = df_hr['reserve_datetime'].dt.date
        df_hr['reserve_hour'] = df_hr['reserve_datetime'].dt.hour
        df_hr['visit_date'] = df_hr['visit_datetime'].dt.date
        df_hr['visit_hour'] = df_hr['visit_datetime'].dt.hour
        df_hr['diff_day'] = df_hr['visit_datetime'].sub(df_hr['reserve_datetime'], axis=0).dt.days
        df_hr['diff_hour'] = (df_hr['visit_datetime'].sub(df_hr['reserve_datetime'], axis=0)).astype('timedelta64[h]')
        print(df_hr)

        df_hr_all_visits_reserve = (df_hr.groupby('visit_date')
           .agg({'hpg_store_id':'count', 'reserve_visitors': 'sum'})
           .reset_index()
           .rename(columns={'hpg_store_id':'hpg_store_id_count', 'reserve_visitors':'all_visitors'})
        )
        print("df_hr_all_visits_reserve", df_hr_all_visits_reserve)
        plot_all_hpg_visits_reserve = (ggplot(df_hr_all_visits_reserve)
            + geom_line(aes(x='visit_date', y='all_visitors', group = 1), color='red')
            + scale_x_datetime(breaks=date_breaks('3 months'))
            + labs(x = "'hpg' visit date")
             )
        self.save_plot(plot_all_hpg_visits_reserve, "plot_all_hpg_visits_reserve", self.dir_plot, 10, 4)

        df_hr_all_visits_reserve_hour = (df_hr.groupby('visit_hour')
           .agg({'hpg_store_id':'count', 'reserve_visitors': 'sum'})
           .reset_index()
           .rename(columns={'hpg_store_id':'hpg_store_id_count', 'reserve_visitors':'all_visitors'})
        )
        plot_all_hpg_visits_reserve_hour = (ggplot(df_hr_all_visits_reserve_hour, aes(x='visit_hour', y='all_visitors'))
            + geom_col(fill="red")
             )
        self.save_plot(plot_all_hpg_visits_reserve_hour, "plot_all_hpg_visits_reserve_hour", self.dir_plot, 10, 4)

        df_hr_all_visits_reserve_diff = (df_hr.groupby('diff_hour')
           .agg({'hpg_store_id':'count', 'reserve_visitors': 'sum'})
           .reset_index()
           .rename(columns={'hpg_store_id':'air_store_id_count', 'reserve_visitors':'all_visitors'})
        )
        df_hr_all_visits_reserve_diff = df_hr_all_visits_reserve_diff.loc[df_hr_all_visits_reserve_diff['diff_hour'] < 24*5]
        print(df_hr_all_visits_reserve_diff)
        plot_all_hpg_visits_reserve_diff = (ggplot(df_hr_all_visits_reserve_diff, aes(x='diff_hour', y='all_visitors'))
            + geom_col(fill = "red")
            + labs(x = "Time from reservation to visit [hours]")
             )
        self.save_plot(plot_all_hpg_visits_reserve_diff, "plot_all_hpg_visits_reserve_diff", self.dir_plot, 10, 4)


    # ============ Step 00_d: Visualize Air Store clusters  ==============
    def visualize_air_store(self):
        df_asi = self.data['asi']

        # Create a Map instance
        m = folium.Map(location=(df_asi['latitude'].mean(), df_asi['longitude'].mean()),
                       tiles='CartoDB positron',
                       attr="CartoDB positron",
                       zoom_start=6)

        mc = MarkerCluster()
        for lat, lon, genre in zip(df_asi['latitude'], df_asi['longitude'],
                            df_asi['air_genre_name']):
            mc.add_child(folium.Marker(location=[lat, lon],
                popup=genre,
                radius=8))
        m.add_child(mc)
        m.save('map_asi.html')


        df_asi_gerne_res = (df_asi.groupby('air_genre_name')
            .size()
            .reset_index(name='counts')
            .sort_values('counts', ascending=False)
        )
        df_asi_gerne_res = df_asi_gerne_res.reset_index(drop=True)
        plot_df_asi_gerne_res = (ggplot(df_asi_gerne_res)
            + aes(x='air_genre_name', weight='counts', fill='air_genre_name')
            + geom_bar()
            + coord_flip()
            + labs(x = "Type of cuisine (air_genre_name)", y = "Number of air restaurants")
             )
        self.save_plot(plot_df_asi_gerne_res, "plot_df_asi_gerne_res", self.dir_plot, 10, 4)

        N = 3
        df_asi_gerne_res = (df_asi_gerne_res.groupby("air_genre_name")
            .sum()
            .sort_values(by='counts', ascending=False)
            .head(N)
            .reset_index()
        )

        print(df_asi_gerne_res)
        plot_df_asi_gerne_res_top3 = (ggplot(df_asi_gerne_res)
            + aes(x='air_genre_name', weight='counts', fill='air_genre_name')
            + geom_bar()
            + coord_flip()
            + labs(x = "Top 3 types of cuisine (air_genre_name)", y = "Number of air restaurants")
             )
        self.save_plot(plot_df_asi_gerne_res_top3, "plot_df_asi_gerne_res_top3", self.dir_plot, 10, 4)

        M=10
        df_asi_area_res = (df_asi.groupby('air_area_name')
            .size()
            .reset_index(name='counts')
            .sort_values('counts', ascending=False)
            .head(M)
        )
        print(df_asi_area_res)
        plot_df_asi_area_res = (ggplot(df_asi_area_res)
            + aes(x='air_area_name', weight='counts')
            + geom_bar()
            + coord_flip()
            + labs(x = "Types of cuisine (air_area_name)", y = "Number of air restaurants")
             )
        self.save_plot(plot_df_asi_area_res, "plot_df_asi_area_res", self.dir_plot, 10, 4)


    # ============ Step 00_e: Visualize HPG Store clusters  ==============
    def visualize_hpg_store(self):
        df_hsi = self.data['hsi']

        # Create a Map instance
        m = folium.Map(location=(df_hsi['latitude'].mean(), df_hsi['longitude'].mean()),
                       tiles='CartoDB positron',
                       attr="CartoDB positron",
                       zoom_start=6)

        mc = MarkerCluster()
        for lat, lon, genre in zip(df_hsi['latitude'], df_hsi['longitude'],
                            df_hsi['hpg_genre_name']):
            mc.add_child(folium.Marker(location=[lat, lon],
                popup=genre,
                radius=8))
        m.add_child(mc)
        m.save('map_hsi.html')


        df_hsi_gerne_res = (df_hsi.groupby('hpg_genre_name')
            .size()
            .reset_index(name='counts')
            .sort_values('counts', ascending=False)
        )
        df_hsi_gerne_res = df_hsi_gerne_res.reset_index(drop=True)
        plot_df_hsi_gerne_res = (ggplot(df_hsi_gerne_res)
            + aes(x='hpg_genre_name', weight='counts', fill='hpg_genre_name')
            + geom_bar()
            + coord_flip()
            + labs(x = "Type of cuisine (hpg_genre_name)", y = "Number of hpg restaurants")
             )
        self.save_plot(plot_df_hsi_gerne_res, "plot_df_hsi_gerne_res", self.dir_plot, 10, 4)

        N = 3
        df_hsi_gerne_res = (df_hsi_gerne_res.groupby("hpg_genre_name")
            .sum()
            .sort_values(by='counts', ascending=False)
            .head(N)
            .reset_index()
        )

        print(df_hsi_gerne_res)
        plot_df_asi_gerne_res_top3 = (ggplot(df_hsi_gerne_res)
            + aes(x='hpg_genre_name', weight='counts', fill='hpg_genre_name')
            + geom_bar()
            + coord_flip()
            + labs(x = "Top 3 types of cuisine (hpg_genre_name)", y = "Number of hpg restaurants")
             )
        self.save_plot(plot_df_asi_gerne_res_top3, "plot_df_asi_gerne_res_top3", self.dir_plot, 10, 4)

        M=10
        df_hsi_area_res = (df_hsi.groupby('hpg_genre_name')
            .size()
            .reset_index(name='counts')
            .sort_values('counts', ascending=False)
            .head(M)
        )
        print(df_hsi_area_res)
        plot_df_hsi_area_res = (ggplot(df_hsi_area_res)
            + aes(x='hpg_genre_name', weight='counts')
            + geom_bar()
            + coord_flip()
            + labs(x = "Types of cuisine (hpg_genre_name)", y = "Number of hpg restaurants")
             )
        self.save_plot(plot_df_hsi_area_res, "plot_df_hsi_area_res", self.dir_plot, 10, 4)

vi = VisualizeInput()
vi.visualize_air_visit()
vi.visualize_air_reserve()
vi.visualize_hpg_reserve()
vi.visualize_air_store()
vi.visualize_hpg_store()
