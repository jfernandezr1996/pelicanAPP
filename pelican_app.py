# Librerias
import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
import seaborn as sns
import random
import base64
sns.set_style("ticks")

np.random.seed(66)
from functions import *

def load_data(ruta, encoding):
    return pd.read_csv(ruta, sep=";", encoding=encoding)

def color_percentiles(val):
    if val < 20:
        color = 'red'
    elif val < 40:
        color = 'coral'
    elif val < 60:
        color = 'gold'
    elif val < 80:
        color = 'lightgreen'
    else:
        color = 'forestgreen'
    return 'color: %s' % color

def color_background_radar_stats(x):
    palette = sns.color_palette(n_colors = len(x))
    colors_radar = palette.as_hex()
    is_max = x == x.max()
    lista = []
    for c,b in zip(colors_radar, is_max):
        if b:
            lista.append('background-color: {}'.format(c))
        else:
            lista.append('')
    return lista

def main():
    
    st.set_page_config(layout="wide")
    
    df = load_data('data/FBREF_clubes.csv', encoding='latin')
    df_description_metrics = load_data('data/summarize_field.csv', encoding='latin')
    df_description_metrics.index = df_description_metrics['Original']
    dict_description_metrics = df_description_metrics['Description'].to_dict()
    
    st.sidebar.title('Pelican Search :soccer:')
    
    menu = st.sidebar.selectbox('Menu', 
                                ['Performance', 
                                 'Ranking', 
                                 'Scatter Plot',
                                 'Similarity Tool'])
    
    if menu == 'Performance':
                
        teams = list(df['Squad'])
        competitions = list(df['Competition'].unique())
        
        with st.sidebar.beta_expander('Filters'):
            competition_selection = st.selectbox(
                    'Competitions', 
                    ['All Competitions'] + competitions)
        
            if competition_selection != 'All Competitions':
                default_dict = {'La Liga': ['Barcelona', 'Real Madrid'], 
                                'Premier League': ['Liverpool', 'Manchester City'], 
                                'Ligue 1': ['PSG', 'Lyon'], 
                                'Bundesliga': ['Bayern Munich', 'Dortmund'], 
                                'Serie A': ['Juventus', 'Inter']}
                teams_competition = list(
                        df[df.Competition == competition_selection]['Squad'])
                default_teams = default_dict[competition_selection]
            else:
                teams_competition = teams
                default_teams = ['Bayern Munich', 'PSG']
            
            teams_selection = st.multiselect(
                    'Teams', options = teams_competition, 
                    default = default_teams)
        
            if competition_selection != 'All Competitions':
                df_competition = df[df.Competition == competition_selection]
            else:
                df_competition = df.copy()
                
        with st.sidebar.beta_expander("Metrics"):
            st.markdown(get_table_download_link('data/description_field.csv'), 
                        unsafe_allow_html=True)
            cols_selection = st.multiselect(
                'Metrics', options = list(df.columns[2:]), 
                default = ['Gls/90', 'Ast/90', 'KP/90',
                       'Press%', 'Passes%', 'Tkl/90', 'Recov/90',
                       'GA/90', 'PSxG+/-', 'SoTA/GA', 'Aerial%', 'Dribbles%', 
                       'Touches/90', 'SoT/G'])

        
        with st.sidebar.beta_expander("Radar Options"):
            radio_radar = st.radio(
                'Radar', ['Traditional Radar', 'Percentile Radar'])
            if radio_radar == 'Percentile Radar':
                average = False
                percentile = True
                ranges_cols = {c: (0,100) for c in df_competition.columns[2:]}            
            else:
                average = st.checkbox('Average Competition')
                percentile = False
                
                ranges_cols = {c: (np.percentile(df_competition[c].values, 5), 
                                   np.percentile(df_competition[c].values, 95)) 
                               for c in df_competition.columns[2:]}
        with st.sidebar.beta_expander("Table Options"):
                select_info_1 = st.checkbox('Full metric name') 

        c1, c2 = st.beta_columns((1, 1))
        plot_radar, df_radar, colors_radar = radarchart_pyplot(
                df, df_competition, 
                teams_selection, cols_selection, 
                ranges_cols, percentile, average,(6,6))
        c1.pyplot(plot_radar)
            
        stats_html = calculate_stats(
                df_radar, df_competition, cols_selection, teams_selection, 
                select_info_1, dict_description_metrics, percentile)
        
        with c2:
            raw_stats_html = stats_html._repr_html_()
            stats = components.html(raw_stats_html, height=500, scrolling=True)

            
    else:
        
        if menu == 'Ranking':
            
            competitions = list(df['Competition'].unique())
            
            with st.sidebar.beta_expander('Filters'):
        
                competition_selection = st.selectbox(
                    'Competitions', ['All Competitions'] + competitions)
            
                if competition_selection != 'All Competitions':
                    default_dict = {'La Liga': ['Barcelona', 'Real Madrid'], 
                                    'Premier League': ['Liverpool', 'Manchester City'], 
                                    'Ligue 1': ['PSG', 'Lyon'], 
                                    'Bundesliga': ['Bayern Munich', 'Dortmund'], 
                                    'Serie A': ['Juventus', 'Inter']}
                    teams = list(df[df.Competition == competition_selection]['Squad'])
                    default_teams = default_dict[competition_selection]

                else:
                    teams = list(df['Squad'])
                    default_teams = ['Barcelona', 'Bayern Munich']
            
                teams_swarmplot = st.multiselect(
                    'Teams', options = teams, default = default_teams)
                metrics_selection = st.multiselect(
                    'Metrics', options = list(df.columns[2:]), 
                    default = ['G-PK/90', 'SoT/90', 'SoTA/90', 
                           'PassesProgressive/90', 'KP/90', 'Aerial%'])
            if len(metrics_selection) < 10:
                st.pyplot(swarmplot_diagram(
                    df, teams_swarmplot, metrics_selection, 
                    dict_description_metrics, competition_selection, 'Squad'))
                
        else:
            
            if menu == 'Similarity Tool':
                
                with st.sidebar.beta_expander('Filters'):
                
                    team_similar = st.selectbox('Team', df.Squad)
                    df_team_similar = df[df.Squad == team_similar]
                
                    competitions_similar = st.multiselect(
                        'Competitions', 
                        options = list(df.Competition.unique()), 
                        default = ['La Liga', 'Premier League', 
                                   'Bundesliga', 'Serie A', 'Ligue 1'])
                
                    df_similar_comp = df[df.Competition.isin(competitions_similar)]
                    df_similar_comp = pd.concat(
                        [df_team_similar, df_similar_comp], axis=0)
                    df_similar_comp.drop_duplicates(inplace=True)
                    df_similar_comp.reset_index(drop=True, inplace=True)
                
                with st.sidebar.beta_expander("Search Options"):
                    option_similar = st.checkbox('Auto Search') 
                
                    if not option_similar:
                        n_similars = st.slider(
                            'Number of similar teams', 1, 10, 4)
                        teams_similar_auto = None
                    else:
                        pos_team = df[df.Squad == team_similar].index.values[0]
                        teams_similar_auto = st.multiselect(
                                'Teams', options = df_similar_comp['Squad'].tolist(),
                                default = df.iloc[pos_team+1,0])
                        n_similars = None
                    
                with st.sidebar.beta_expander("Metrics"):
                    metrics_selection = st.selectbox(
                        'Metrics Selection', ['Filtered', 'All metrics'])
                    
                    if metrics_selection == 'Filtered':
                        metrics_similar = st.multiselect(
                            '', options = list(df_similar_comp.columns[2:]), 
                            default = ['Gls/90', 'SoT/G', 'PSxG+/-',
                                       'Ast/90', 'GCA/90', 'KP/90', 'PPA/90', 
                                       'Press%', 'Passes%', 'Poss%', 'Tkl/90', 
                                       'Aerial%', 'xGA/90', 'Err/90', 'Fls/90', 
                                       'Dribbles%'])
                        df_metrics_filtered = df_similar_comp[['Squad','Competition'] \
                                                              + metrics_similar]
                    else:
                        metrics_radar = st.multiselect(
                            'Metrics Radar', 
                            options = list(df_similar_comp.columns[2:]), 
                            default = ['Gls/90', 'SoT/G', 'PSxG+/-',
                                       'Ast/90', 'GCA/90', 'KP/90', 'PPA/90', 
                                       'Press%', 'Passes%', 'Poss%', 'Tkl/90', 
                                       'Aerial%', 'xGA/90', 'Fls/90', 
                                       'Dribbles%'])
                        metrics_similar = list(df_similar_comp.columns[2:])
                        df_metrics_filtered = df_similar_comp
                        
                    
                df_similar_t, html_tabla = similar_team(
                     df_metrics_filtered, team_similar, n_similars, 
                     teams_similar_auto, option_similar)
                
                if not option_similar:
                    title = 'Most Similar Teams'
                else:
                    title = 'Finding Similar Teams'
                    
                c1, c2 = st.beta_columns((1,1.25))
                with c1:
                    team_img = 'teams/'+team_similar+".png"
                    st.markdown("""
                                <style>
                                .container {
                                    display: flex;
                                    }
                                .logo-text {
                                    font-weight:500;
                                    font-family:IBM Plex Sans;
                                    font-size:30px;
                                    padding-top:10px;
                                    }
                                .logo-img {
                                    float: right;
                                    width: 20%;
                                    height: 80px;
                                    }
                                </style>""", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="container">
                        <p class="logo-text">Similarity Tool &nbsp</p>
                        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(team_img, "rb").read()).decode()}">
                        </div>
                        """,
                        unsafe_allow_html=True)
                    st.subheader(title)
                    raw_html = html_tabla._repr_html_()
                    components.html(raw_html, height=500, scrolling=True)
                df_similar_t = df_similar_t.sort_values(
                    by='% Similarity', ascending=False)
                teams_radar = df_similar_t['Squad'].tolist()
                df_plot_similar = df_metrics_filtered[
                    df_metrics_filtered.Squad.isin(teams_radar)]
                df_plot_similar.reset_index(drop=True, inplace=True)
                
                ranges = {c: (np.percentile(df_metrics_filtered[c].values, 5), 
                              np.percentile(df_metrics_filtered[c].values, 95)) 
                          for c in df_metrics_filtered.columns[2:]}
                
                if metrics_selection != 'Filtered':
                    random_metrics = metrics_radar
                    ranges_metrics = dict()
                    for i in random_metrics:
                        ranges_metrics[i] = ranges[i]
                else:
                    random_metrics = metrics_similar
                    ranges_metrics = ranges
                plot_radar_similar, _, _ = radarchart_pyplot(
                    df, df_plot_similar, df_plot_similar['Squad'].tolist(), 
                    random_metrics, ranges_metrics, False, False,(4,4))
                c2.pyplot(plot_radar_similar)
                
            else:
            
                if menu == 'Scatter Plot':
                    
                    competitions = list(df['Competition'].unique())
                    
                    with st.sidebar.beta_expander('Filters'):
            
                        competition = st.multiselect(
                            'Competitions', options = competitions, default = competitions)
                        
                        dashboard = st.selectbox(
                            'Dashboard', ['Predefined', 'Customized'])
                    
                        categories = ['All Stats', 'Standard Stats', 'Goalkeeping', 'Shooting', 
                                      'Passing', 'Goal and Shot Creation', 'Defensive Actions', 
                                      'Possession', 'Miscellaneous Stats']
            
                        category_dict = {'Standard Stats': 
                                         ['Poss%', 'Gls', 'Ast', 'xG', 'xA', 
                                          'Gls/90', 'Ast/90', 'xG/90', 'xA/90', 
                                          'SoT/G', 'SoTA/GA'], 
                                         'Goalkeeping': 
                                             ['GA', 'OG', 'GA/90', 'Save%', 'CS%', 
                                              'SoTA/90', 'PSxG', 'PSxG/SoT', 'xGA/90', 
                                              'PSxG+/-', 'SoTA/GA'],
                                         'Shooting': ['Gls', 'Gls/90', 
                                                      'Sh', 'Sh/90', 'SoT/90', 'xG', 
                                                      'xG/90', 'SoT/G', 'G-xG'],
                                         'Passing': ['Passes%', 'ShortPasses%', 'MediumPasses%', 
                                                     'LongPasses%', 'PassesCompleted/90', 
                                                     'PassesAttempted/90', 'ShortPassesCompleted/90', 
                                                     'MediumPassesCompleted/90', 
                                                     'LongPassesCompleted/90', 
                                                     'TotDistPasses/90',
                                                     'PrgDistPasses/90','KP/90', 
                                                     'FinalThirdPasses/90', 'PPA/90', 
                                                     'CrsPA/90'], 
                                         'Goal and Shot Creation': ['Gls/90', 'Sh/90',
                                                                    'SCA/90', 'GCA/90'], 
                                         'Defensive Actions': ['Press%', 'Tkl/90', 'Blocks/90', 
                                                               'Int/90', 'Err/90',], 
                                         'Possession': ['Poss%', 'Touches/90', 'Dribbles%', 
                                                        'Gls/90', 'Points/90'],
                                         'Playing Time': ['xG/90', 'xGA/90', 'Points/90'],
                                         'Miscellaneous Stats': ['Aerial%', 'Fls/90', 
                                                             'Int/90', 'Tkl/90', 'Recov/90']}
                        
                        customized_stats = {'Standard Stats': ['SoT/G', 'SoTA/GA'], 
                                            'Goalkeeping': ['GA', 'PSxG+/-'], 
                                            'Shooting': ['Gls', 'G-xG'], 
                                            'Passing': ['FinalThirdPasses/90', 'KP/90'], 
                                            'Goal and Shot Creation': ['GCA/90', 'Gls/90'], 
                                            'Defensive Actions': ['Press%', 'Int/90'], 
                                            'Possession': ['Poss%', 'Points/90'], 
                                            'Playing Time': ['Points/90', 'xG/90'],
                                            'Miscellaneous Stats': ['Fls/90', 'Tkl/90']}
                        
                    if dashboard == 'Customized':
                        with st.sidebar.beta_expander('Metrics'):
                            category_stats = st.selectbox('Category', categories)
                            if category_stats == 'All Stats':
                                stats_selection = list(df.columns[2:])
                            else:
                                stats_selection = category_dict[category_stats]
                            stats = st.multiselect('Metrics', options = stats_selection)
                            
                        if len(stats) == 2:

                            with st.sidebar.beta_expander("Graphic Options"):
                                select_tendency = st.checkbox('View Linear Trendline')
                            with st.sidebar.beta_expander("Table Options"):
                                info = st.checkbox('Full metric name')
                                sort = st.selectbox('Sort by', stats)
                                
                            st.plotly_chart(
                                scatterplot_stats(df, stats, competition, 
                                                  dict_description_metrics, 
                                                  select_tendency, "", "", False), 
                                use_container_width=True)
                            
                            html_scatter = table_stats_scatter(
                                df, stats, competition, 
                                dict_description_metrics, info, sort)
                            raw_scatter = html_scatter._repr_html_()
                            components.html(raw_scatter, height=500, scrolling=True)
                    else:
                        
                        with st.sidebar.beta_expander('Metrics'):
                        
                            category_stats = st.selectbox('Category', categories[1:])
                        
                            colors = {'Standard Stats': ['darkorange', 'red', 'green', 'gold'], 
                                      'Goalkeeping': ['darkorange', 'red', 'green', 'gold'], 
                                      'Shooting': ['red', 'darkorange', 'gold', 'green'], 
                                      'Passing': ['red', 'darkorange', 'gold', 'green'],
                                      'Goal and Shot Creation': ['red', 'darkorange', 'gold', 'green'],
                                      'Defensive Actions': ['red', 'darkorange', 'gold', 'green'],
                                      'Possession': ['red', 'darkorange', 'gold', 'green'], 
                                      'Miscellaneous Stats': ['darkorange', 'red', 'green', 'gold']}
                        
                            minus_values = {'Standard Stats': 0.2, 
                                            'Goalkeeping': 2,
                                            'Shooting': 2, 
                                            'Passing': 1,
                                            'Goal and Shot Creation': 0.2,
                                            'Defensive Actions': 0.5,
                                            'Possession': 0.5, 
                                            'Miscellaneous Stats': 0.5}
                        
                        with st.sidebar.beta_expander("Graphic Options"):
                            select_tendency = st.checkbox('View Linear Trendline')
                        with st.sidebar.beta_expander("Table Options"):
                            info = st.checkbox('Full metric name')
                            sort = st.selectbox(
                                'Sort table descending by', 
                                customized_stats[category_stats])
                            
                        st.plotly_chart(scatterplot_stats(
                                df, customized_stats[category_stats], 
                                competition, 
                                dict_description_metrics, select_tendency, 
                                colors[category_stats], minus_values[category_stats], True), 
                                use_container_width=True)
                        
                        html_scatter = table_stats_scatter(
                                df, customized_stats[category_stats], competition, 
                                dict_description_metrics, info, sort)
                        raw_scatter = html_scatter._repr_html_()
                        components.html(raw_scatter, height=500, scrolling=True)

                    
                
        
        
        
        
        
                    
            
if __name__ == "__main__":
    main()