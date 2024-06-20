# -*- coding: utf-8 -*-
"""
112學年度大三學生學習經驗調查
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import re
import seaborn as sns
import streamlit as st 
import streamlit.components.v1 as stc 
#os.chdir(r'C:\Users\user\Dropbox\系務\校務研究IR\大一新生學習適應調查分析\112')

####### 資料前處理
## see file "Junior_Opinions_2.py"

####### 定義相關函數 (Part 1)
###### 載入資料
@st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def load_data(path):
    df = pd.read_pickle(path)
    return df

###### 計算次數分配並形成 包含'項目', '人數', '比例' 欄位的 dataframe 'result_df'
# @st.cache_data(ttl=3600, show_spinner="正在處理資料...")  ## Add the caching decorator
# def Frequency_Distribution(df, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1,row_rank=False, row_rank_number=3): ## 當有去掉dropped_string & 是單選題時, sum_choice 要使用 0
def Frequency_Distribution(df, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1): ## 當有去掉dropped_string & 是單選題時, sum_choice 要使用 0
    ##### 去掉df在指定的column 'column_index' 中包含 NaN 的 所有rows 並付值給df_restrict. df本身直接去掉會出現問題, 原因不明 ?
    # df.dropna(subset=[df.columns[column_index]], inplace=True)
    # df=df_junior
    df_restrict = df.dropna(subset=[df.columns[column_index]])

    # if row_rank==True:
    #     ##### 使用 str.split 方法分割第14行的字串，以 ';' 為分隔符, 然後使用 apply 和 lambda 函數來提取前三個元素, 並再度以;分隔.
    #     # df_junior['col14'] = df_junior['col14'].str.split(';').apply(lambda x: ';'.join(x[:3]))
    #     df.iloc[:,column_index] = df.iloc[:,column_index].str.split(split_symbol).apply(lambda x: ';'.join(x[:row_rank_number]))

    ##### 将字符串按split_symbol分割并展平以及前處理
    # if row_rank==True:
    #     # split_values = df.iloc[:,column_index].str.split(split_symbol).apply(lambda x: ';'.join(x[:row_rank_number])).explode()
    #     split_values = df.iloc[:,column_index].str.split(split_symbol).apply(lambda x: x[:row_rank_number]).explode()
    # else:
    #     split_values = df.iloc[:,column_index].str.split(split_symbol).explode()  ## split_symbol=';'

    split_values = df_restrict.iloc[:,column_index].str.split(split_symbol).explode()  ## split_symbol=';' 
    # split_values = df.iloc[:,column_index].str.split(split_symbol).explode()
    # type(split_values)  ## pandas.core.series.Series
    # print(split_values)
    # '''
    # 0            志工服務
    # 0            校外實習
    # 0            企業參訪
    # 0      擔任社團/系學會幹部
    # 0            參加社團
       
    # 876    擔任社團/系學會幹部
    # 876          志工服務
    # 876          校外實習
    # 876          企業參訪
    # 876              
    # Name: 您認為下列哪些經驗對未來工作會有所幫助？(可複選), Length: 157, dtype: object
    # '''
    
    #### split_values資料前處理
    ### 去掉每一個字串前後的space
    split_values = split_values.str.strip()
    # ### 去掉每一個字串最後的;符號
    # split_values = split_values.str.rstrip(';')
    ### 使用 dropna 方法去掉index或值為 NA 或 NaN 的項目
    split_values = split_values.dropna()
    # ### 查看 split_values中相異的資料: 有空字串
    # split_values.unique()
    # '''
    # array(['志工服務', '校外實習', '企業參訪', '擔任社團/系學會幹部', '參加社團', '', '不確定', '沒幫助'],
    #       dtype=object)
    # '''
    ### 使用布林索引去掉值為空字串的資料
    split_values = split_values[split_values != '']
    # ### 查看 split_values中相異的資料: 空字串不見了
    # split_values.unique()
    # '''
    # array(['志工服務', '校外實習', '企業參訪', '擔任社團/系學會幹部', '參加社團', '不確定', '沒幫助'],
    #       dtype=object)
    # '''
    ### 將以 '其他' 開頭的字串簡化為 '其他'; 
    ## <注意> np.where 的邏輯中，NaN/NA 不被視為 False，而是默認處理為 True, 因此，np.where 將 NaN/NA 視為符合條件，並將其替換為 '其他'。
    ## 如果不希望 NaN/NA 值被替換為 '其他'，可以在使用 np.where 之前明確地將 NaN 值處理為 False，或者在替換時保留 NaN 值。
    split_values_np = np.where(split_values.str.startswith('其他').fillna(False), '其他', split_values)
    split_values = pd.Series(split_values_np)  ## 轉換為 pandas.core.series.Series

    ##### 计算不同子字符串的出现次数以及前處理
    value_counts = split_values.value_counts()
    # type(value_counts)  ## pandas.core.series.Series
    # print(value_counts)
    # #### split_values 沒有去掉空字串之前結果:
    # '''
    #               40
    # 校外實習          35
    # 企業參訪          27
    # 擔任社團/系學會幹部    22
    # 志工服務          15
    # 參加社團          12
    # 不確定            5
    # 沒幫助            1
    # Name: 您認為下列哪些經驗對未來工作會有所幫助？(可複選), dtype: int64
    # '''
    # #### split_values 去掉空字串之後結果:
    # '''
    # 校外實習          35
    # 企業參訪          27
    # 擔任社團/系學會幹部    22
    # 志工服務          15
    # 參加社團          12
    # 不確定            5
    # 沒幫助            1
    # Name: 您認為下列哪些經驗對未來工作會有所幫助？(可複選), dtype: int64
    # '''
    
    #### 去掉 '沒有工讀' index的值:
    if dropped_string in value_counts.index:
        value_counts = value_counts.drop(dropped_string)
    #### 使用 dropna 方法去掉index或值為 NA 或 NaN 的項目
    value_counts = value_counts.dropna()
        
    ##### 計算總數方式的選擇:
    if sum_choice == 0:    ## 以 "人次" 計算總數; 但如果是單選題, 此選項即為填答人數, 並且會去掉填答 "dropped_string" 以及有NA項目(index或值)的人數, 例如 dropped_string='沒有工讀'.
        total_sum = value_counts.sum()
    if sum_choice == 1:    ## 以 "填答人數" 計算總數
        total_sum = df.shape[0]
        
    ##### 计算不同子字符串的比例
    # proportions = value_counts/value_counts.sum()
    proportions = value_counts/total_sum
    
    ##### 轉化為 numpy array
    value_counts_numpy = value_counts.values
    proportions_numpy = proportions.values
    items_numpy = proportions.index.to_numpy()
    
    ##### 创建一个新的DataFrame来显示结果
    result_df = pd.DataFrame({'項目':items_numpy, '人數': value_counts_numpy,'比例': proportions_numpy.round(4)})
    return result_df


# @st.cache_data(ttl=3600, show_spinner="正在處理資料...")  ## Add the caching decorator
# def Frequency_Distribution(df, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1):
#     ##### 将字符串按逗号分割并展平
#     split_values = df.iloc[:,column_index].str.split(split_symbol).explode()  ## split_symbol=','
#     #### split_values資料前處理
#     ### 去掉每一個字串前後的space
#     split_values = split_values.str.strip()
#     ### 將以 '其他' 開頭的字串簡化為 '其他'
#     split_values_np = np.where(split_values.str.startswith('其他'), '其他', split_values)
#     split_values = pd.Series(split_values_np)  ## 轉換為 pandas.core.series.Series
    
#     ##### 计算不同子字符串的出现次数
#     value_counts = split_values.value_counts()
#     #### 去掉 '沒有工讀' index的值:
#     if dropped_string in value_counts.index:
#         value_counts = value_counts.drop(dropped_string)
        
#     ##### 計算總數方式的選擇:
#     if sum_choice == 0:    ## 以 "人次" 計算總數
#         total_sum = value_counts.sum()
#     if sum_choice == 1:    ## 以 "填答人數" 計算總數
#         total_sum = df.shape[0]
        
#     ##### 计算不同子字符串的比例
#     # proportions = value_counts/value_counts.sum()
#     proportions = value_counts/total_sum
    
#     ##### 轉化為 numpy array
#     value_counts_numpy = value_counts.values
#     proportions_numpy = proportions.values
#     items_numpy = proportions.index.to_numpy()
    
#     ##### 创建一个新的DataFrame来显示结果
#     result_df = pd.DataFrame({'項目':items_numpy, '人數': value_counts_numpy,'比例': proportions_numpy.round(4)})
#     return result_df






###### 調整項目次序
##### 函数：调整 DataFrame 以包含所有項目(以下df['項目']與order的聯集, 實際應用時, df['項目']是order的子集)，且顺序正确(按照以下的order)
# @st.cache_data(ttl=3600, show_spinner="正在加載資料...")  ## Add the caching decorator
def adjust_df(df, order):
    # 确保 DataFrame 包含所有滿意度值
    for item in order:
        if item not in df['項目'].values:
            # 创建一个新的 DataFrame，用于添加新的row
            new_row = pd.DataFrame({'項目': [item], '人數': [0], '比例': [0]})
            # 使用 concat() 合并原始 DataFrame 和新的 DataFrame
            df = pd.concat([df, new_row], ignore_index=True)

    # 根据期望的顺序重新排列 DataFrame
    df = df.set_index('項目').reindex(order).reset_index()
    return df











#######  读取Pickle文件
df_junior_original = load_data('df_junior_original.pkl')
# df_junior_original = load_data(r'C:\Users\user\Dropbox\系務\校務研究IR\大一新生學習適應調查分析\112\GitHub上傳\df_junior_original.pkl')
###### 使用rename方法更改column名称: '學系' -> '科系'
df_junior_original = df_junior_original.rename(columns={'學系': '科系'})
# ###### 更改院的名稱: 理學->理學院, 資訊->資訊學院, 管理->管理學院, 人社->人文暨社會科學院, 國際->國際學院, 外語->外語學院
# ##### 定义替换规则
# replace_rules = {
#     '理學': '理學院',
#     '資訊': '資訊學院',
#     '管理': '管理學院',
#     '人社': '人文暨社會科學院',
#     '國際': '國際學院',
#     '外語': '外語學院'
# }
# ##### 应用替换规则
# df_junior_original['學院'] = df_junior_original['學院'].replace(replace_rules)
df_ID = load_data('df_ID.pkl')


# df_junior_original.dropna(subset=[df_junior_original.columns[14]], inplace=True)
# df_junior_original.iloc[:,14] = df_junior_original.iloc[:,14].str.split(';').apply(lambda x: ';'.join(x[:1]))
# choice='大傳系' ##'化科系'
# df_junior = df_junior_original[df_junior_original['科系']==choice]
# df_junior.dropna(subset=[df_junior.columns[16]], inplace=True)
# # df_junior.iloc[:,14] = df_junior.iloc[:,14].str.split(';').apply(lambda x: ';'.join(x[:1]))
# df_junior.iloc[:,16].str.split(';').apply(lambda x: ';'.join(x[:2]))


####### 預先設定
global 系_院_校, choice, df_junior, choice_faculty, df_junior_faculty, selected_options, collections, column_index, dataframes, desired_order, combined_df
###### 預設定院或系之選擇
系_院_校 = '0'
###### 預設定 df_junior 以防止在等待選擇院系輸入時, 發生後面程式df_junior讀不到資料而產生錯誤
choice='財金系' ##'化科系'
df_junior = df_junior_original[df_junior_original['科系']==choice]
# choice_faculty = df_junior['學院'][0]  ## 選擇學系所屬學院: '理學院'
choice_faculty = df_junior['學院'].values[0]  ## 選擇學系所屬學院: '理學院'
df_junior_faculty = df_junior_original[df_junior_original['學院']==choice_faculty]  ## 挑出全校所屬學院之資料
# df_junior_faculty['學院']  

###### 預設定 selected_options, collections
selected_options = ['化科系','企管系']
# collections = [df_junior_original[df_junior_original['學院']==i] for i in selected_options]
collections = [df_junior_original[df_junior_original['科系']==i] for i in selected_options]
# collections = [df_junior, df_junior_faculty, df_junior_original]
# len(collections) ## 2
# type(collections[0])   ## pandas.core.frame.DataFrame
column_index = 9
dataframes = [Frequency_Distribution(df, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1) for df in collections]  ## 22: "您工讀次要的原因為何:"
# len(dataframes)  ## 2
# len(dataframes[1]) ## 2
# len(dataframes[0]) ## 2


##### 形成所有學系'項目'欄位的所有值
desired_order  = list(set([item for df in dataframes for item in df['項目'].tolist()])) 
# desired_order  = list(set([item for item in dataframes[0]['項目'].tolist()])) 

##### 缺的項目值加以擴充， 並統一一樣的項目次序
dataframes = [adjust_df(df, desired_order) for df in dataframes]
# len(dataframes)  ## 2
# len(dataframes[1]) ## 6
# len(dataframes[0]) ## 6, 從原本的5變成6 
# dataframes[0]['項目']
# '''
# 0              體驗生活
# 1         為未來工作累積經驗
# 2             負擔生活費
# 3              增加人脈
# 4    不須負擔生活費但想增加零用錢
# 5         學習應對與表達能力
# Name: 項目, dtype: object
# '''
# dataframes[1]['項目']
# '''
# 0              體驗生活
# 1         為未來工作累積經驗
# 2             負擔生活費
# 3              增加人脈
# 4    不須負擔生活費但想增加零用錢
# 5         學習應對與表達能力
# Name: 項目, dtype: object
# '''
# global combined_df
combined_df = pd.concat(dataframes, keys=selected_options)
# combined_df = pd.concat(dataframes, keys=[choice,choice_faculty,'全校'])
# ''' 
#                    項目  人數      比例
# 化科系 0            體驗生活   0  0.0000
#     1       為未來工作累積經驗  13  0.3824
#     2           負擔生活費   2  0.0588
#     3            增加人脈   2  0.0588
#     4  不須負擔生活費但想增加零用錢   7  0.2059
#     5       學習應對與表達能力  10  0.2941
# 企管系 0            體驗生活   1  0.0417
#     1       為未來工作累積經驗   9  0.3750
#     2           負擔生活費   4  0.1667
#     3            增加人脈   2  0.0833
#     4  不須負擔生活費但想增加零用錢   2  0.0833
#     5       學習應對與表達能力   6  0.2500
# '''


####### 定義相關函數 (Part 2): 因為函數 'Draw' 的定義需要使用 'dataframes','combined_df' 來進行相關計算, 因此要放在以上 '預先設定' 之後才會有 'dataframes', 'combined_df' 的值
###### 畫圖形(單一學系或學院, 比較圖形)
# @st.cache_data(ttl=3600, show_spinner="正在處理資料...")  ## Add the caching decorator
## 當有去掉dropped_string & 是單選題時, sum_choice 要使用 0
# def Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=pd.DataFrame(), selected_options=[], dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name='', rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order, row_rank=False, row_rank_number=3):
def Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=pd.DataFrame(), selected_options=[], dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name='', rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order):    
    ##### 使用Streamlit畫單一圖
    if 系_院_校 == '0':
        collections = [df_junior, df_junior_faculty, df_junior_school]
        if rank == True:
            # dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice, row_rank, row_rank_number).head(rank_number) for df in collections]  ## 'dataframes' list 中的各dataframe已經是按照次數高至低的項目順序排列
            dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice).head(rank_number) for df in collections]  ## 'dataframes' list 中的各dataframe已經是按照次數高至低的項目順序排列
        else:
            # dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice, row_rank, row_rank_number) for df in collections]  ## 'dataframes' list 中的各dataframe已經是按照次數高至低的項目順序排列
            dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice) for df in collections]  ## 'dataframes' list 中的各dataframe已經是按照次數高至低的項目順序排列
        ## 形成所有學系'項目'欄位的所有值
        # desired_order  = list(set([item for df in dataframes for item in df['項目'].tolist()]))
        # desired_order  = list(set([item for item in dataframes[0]['項目'].tolist()])) 
        #### 只看所選擇學系的項目(已經是按照次數高至低的項目順序排列), 並且反轉次序使得表與圖的項目次序一致
        desired_order  = [item for item in dataframes[0]['項目'].tolist()]  ## 只看所選擇學系的項目
        desired_order = desired_order[::-1]  ## 反轉次序使得表與圖的項目次序一致
        ## 缺的項目值加以擴充， 並統一一樣的項目次序
        dataframes = [adjust_df(df, desired_order) for df in dataframes]
        combined_df = pd.concat(dataframes, keys=[choice,choice_faculty,'全校'])
        # 获取level 0索引的唯一值并保持原始顺序
        unique_level0 = combined_df.index.get_level_values(0).unique()

        #### 設置 matplotlib 支持中文的字體: 
        # matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
        # matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        # matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        #### 设置条形的宽度
        # bar_width = 0.2
        #### 设置y轴的位置
        r = np.arange(len(dataframes[0]))  ## len(result_df_理學_rr)=6, 因為result_df_理學_rr 有 6個 row: 非常滿意, 滿意, 普通, 不滿意, 非常不滿意
        # #### 设置字体大小
        # title_fontsize = title_fontsize ##15
        # xlabel_fontsize = xlabel_fontsize  ##14
        # ylabel_fontsize = ylabel_fontsize  ##14
        # xticklabel_fontsize = 14
        # yticklabel_fontsize = 14
        # annotation_fontsize = 8
        # legend_fontsize = legend_fontsize  ##14
        #### 绘制条形
        fig, ax = plt.subplots(figsize=(width1, heigh1))
        # for i, (college_name, df) in enumerate(combined_df.groupby(level=0)):
        for i, college_name in enumerate(unique_level0):            
            df = combined_df.loc[college_name]
            # 计算当前分组的条形数量
            num_bars = len(df)
            # 生成当前分组的y轴位置
            index = np.arange(num_bars) + i * bar_width
            # index = r + i * bar_width
            rects = ax.barh(index, df['比例'], height=bar_width, label=college_name)
    
            # # 在每个条形上标示比例
            # for rect, ratio in zip(rects, df['比例']):
            #     ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height(), f'{ratio:.1%}', ha='center', va='bottom',fontsize=annotation_fontsize)
        ### 添加图例
        if fontsize_adjust==0:
            ax.legend()
        if fontsize_adjust==1:
            ax.legend(fontsize=legend_fontsize)

        # ### 添加x轴标签
        # ## 计算每个组的中心位置作为x轴刻度位置
        # # group_centers = r + bar_width * (num_colleges / 2 - 0.5)
        # # group_centers = np.arange(len(dataframes[0]))
        # ## 添加x轴标签
        # # ax.set_xticks(group_centers)
        # # dataframes[0]['項目'].values
        # # "array(['個人興趣', '未來能找到好工作', '落點分析', '沒有特定理由', '家人的期望與建議', '師長推薦'],dtype=object)"
        # ax.set_xticks(r + bar_width * (len(dataframes) / 2))
        # ax.set_xticklabels(dataframes[0]['項目'].values, fontsize=xticklabel_fontsize)
        # # ax.set_xticklabels(['非常滿意', '滿意', '普通', '不滿意','非常不滿意'],fontsize=xticklabel_fontsize)
        
        ### 设置x,y轴刻度标签
        ax.set_yticks(r + bar_width*(len(dataframes) / 2))  # 调整位置以使标签居中对齐到每个条形
        if fontsize_adjust==0:
            ax.set_yticklabels(dataframes[0]['項目'].values)
            ax.tick_params(axis='x')
        if fontsize_adjust==1:
            ax.set_yticklabels(dataframes[0]['項目'].values, fontsize=yticklabel_fontsize)
            ## 设置x轴刻度的字体大小
            ax.tick_params(axis='x', labelsize=xticklabel_fontsize)
        # ax.set_yticklabels(dataframes[0]['項目'].values)
        # ax.set_yticklabels(dataframes[0]['項目'].values, fontsize=yticklabel_fontsize)


        ### 设置标题和轴标签
        if fontsize_adjust==0:
            ax.set_title(item_name)
        if fontsize_adjust==1:
            ax.set_title(item_name,fontsize=title_fontsize)
        
        # ax.set_xlabel('满意度',fontsize=xlabel_fontsize)
        if fontsize_adjust==0:
            ax.set_xlabel('比例')
        if fontsize_adjust==1:
            ax.set_xlabel('比例',fontsize=xlabel_fontsize)
        
        ### 显示网格线
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        plt.tight_layout()
        # plt.show()
        ### 在Streamlit中显示
        st.pyplot(plt)

    if 系_院_校 == '1':
    # else:  ## 包含 系_院_校 == '1', 系_院_校 == '2'
        #### 設置中文顯示
        # matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
        # matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        #### 创建图形和坐标轴
        plt.figure(figsize=(width2, heigh2))
        #### 绘制条形图
        ### 反轉 dataframe result_df 的所有行的值的次序,  使得表與圖的項目次序一致
        result_df = result_df.iloc[::-1].reset_index(drop=True)
        if rank == True:
            result_df = result_df.head(rank_number)

        # plt.barh(result_df['項目'], result_df['人數'], label=choice, width=bar_width)
        plt.barh(result_df['項目'], result_df['人數'], label=choice)
        #### 標示比例數據
        for i in range(len(result_df['項目'])):
            if fontsize_adjust==0:
                plt.text(result_df['人數'][i]+1, result_df['項目'][i], f'{result_df.iloc[:, 2][i]:.1%}')
            if fontsize_adjust==1:
                plt.text(result_df['人數'][i]+1, result_df['項目'][i], f'{result_df.iloc[:, 2][i]:.1%}', fontsize=annotation_fontsize)
            
        #### 添加一些图形元素
        if fontsize_adjust==0:
            plt.title(item_name)
            plt.xlabel('人數')
        if fontsize_adjust==1:
            plt.title(item_name, fontsize=title_fontsize)
            plt.xlabel('人數', fontsize=xlabel_fontsize)
        
        #plt.ylabel('本校現在所提供的資源或支援事項')
        #### 调整x轴和y轴刻度标签的字体大小
        if fontsize_adjust==0:
            # plt.tick_params(axis='both')
            ## 设置x轴刻度的字体大小
            plt.tick_params(axis='x')
            ## 设置y轴刻度的字体大小
            plt.tick_params(axis='y')
        if fontsize_adjust==1:
            # plt.tick_params(axis='both', labelsize=xticklabel_fontsize)  # 同时调整x轴和y轴 
            ## 设置x轴刻度的字体大小
            plt.tick_params(axis='x', labelsize=xticklabel_fontsize)
            ## 设置y轴刻度的字体大小
            plt.tick_params(axis='y', labelsize=yticklabel_fontsize)
        
        if fontsize_adjust==0:
            plt.legend()
        if fontsize_adjust==1:
            plt.legend(fontsize=legend_fontsize)
        
        #### 显示网格线
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        #### 显示图形
        ### 一般顯示
        # plt.show()
        ### 在Streamlit中显示
        st.pyplot(plt)
        
    if 系_院_校 == '2':
    # else:  ## 包含 系_院_校 == '1', 系_院_校 == '2'
        #### 設置中文顯示
        # matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
        # matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
        matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        #### 创建图形和坐标轴
        plt.figure(figsize=(width2, heigh2))
        #### 绘制条形图
        ### 反轉 dataframe result_df 的所有行的值的次序,  使得表與圖的項目次序一致
        result_df = result_df.iloc[::-1].reset_index(drop=True)
        if rank == True:
            result_df = result_df.head(rank_number)

        # plt.barh(result_df['項目'], result_df['人數'], label=choice, width=bar_width)
        plt.barh(result_df['項目'], result_df['人數'], label=choice)
        #### 標示比例數據
        for i in range(len(result_df['項目'])):
            if fontsize_adjust==0:
                plt.text(result_df['人數'][i]+1, result_df['項目'][i], f'{result_df.iloc[:, 2][i]:.1%}')
            if fontsize_adjust==1:
                plt.text(result_df['人數'][i]+1, result_df['項目'][i], f'{result_df.iloc[:, 2][i]:.1%}', fontsize=annotation_fontsize)
            
        #### 添加一些图形元素
        if fontsize_adjust==0:
            plt.title(item_name)
            plt.xlabel('人數')
        if fontsize_adjust==1:
            plt.title(item_name, fontsize=title_fontsize)
            plt.xlabel('人數', fontsize=xlabel_fontsize)
        
        #plt.ylabel('本校現在所提供的資源或支援事項')
        #### 调整x轴和y轴刻度标签的字体大小
        if fontsize_adjust==0:
            # plt.tick_params(axis='both')
            ## 设置x轴刻度的字体大小
            plt.tick_params(axis='x')
            ## 设置y轴刻度的字体大小
            plt.tick_params(axis='y')
        if fontsize_adjust==1:
            # plt.tick_params(axis='both', labelsize=xticklabel_fontsize)  # 同时调整x轴和y轴 
            ## 设置x轴刻度的字体大小
            plt.tick_params(axis='x', labelsize=xticklabel_fontsize)
            ## 设置y轴刻度的字体大小
            plt.tick_params(axis='y', labelsize=yticklabel_fontsize)
        
        if fontsize_adjust==0:
            plt.legend()
        if fontsize_adjust==1:
            plt.legend(fontsize=legend_fontsize)
        
        #### 显示网格线
        plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
        #### 显示图形
        ### 一般顯示
        # plt.show()
        ### 在Streamlit中显示
        st.pyplot(plt)



    ##### 使用streamlit 畫比較圖 
    # st.subheader("不同單位比較")    
    # if 系_院_校 == '0' or '1' or '2':
    ## 以下選擇單位要從 df_junior_original 選, 若從df_junior選擇, 就是限定某單位了, 再從此單位去選別單位, 是選不到的.
    if 系_院_校 == '0': 
        collections = [df_junior_school[df_junior_school['科系']==i] for i in selected_options]
        
        if rank == True:
            # dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice, row_rank, row_rank_number).head(rank_number) for df in collections]  ## 'dataframes' list 中的各dataframe已經是按照次數高至低的項目順序排列
            dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice).head(rank_number) for df in collections]  ## 'dataframes' list 中的各dataframe已經是按照次數高至低的項目順序排列
        else:
            # dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice, row_rank, row_rank_number) for df in collections]
            dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice) for df in collections]


        # #### 只看第一個選擇學系的項目(已經是按照次數高至低的項目順序排列), 並且反轉次序使得表與圖的項目次序一致
        # desired_order  = [item for item in dataframes[0]['項目'].tolist()]  ## 只看第一個選擇學系的項目
        # desired_order = desired_order[::-1]  ## 反轉次序使得表與圖的項目次序一致
        ## 形成所有學系'項目'欄位的所有值
        # desired_order  = list(set([item for df in dataframes for item in df['項目'].tolist()])) 
        desired_order  = list(dict.fromkeys([item for df in dataframes for item in df['項目'].tolist()]))
        desired_order = desired_order[::-1]  ## 反轉次序使得表與圖的項目次序一致

        ## 缺的項目值加以擴充， 並統一一樣的項目次序
        dataframes = [adjust_df(df, desired_order) for df in dataframes]
        combined_df = pd.concat(dataframes, keys=selected_options)
    elif 系_院_校 == '1':
        collections = [df_junior_school[df_junior_school['學院']==i] for i in selected_options]
        
        if rank == True:
            # dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice, row_rank, row_rank_number).head(rank_number) for df in collections]  ## 'dataframes' list 中的各dataframe已經是按照次數高至低的項目順序排列
            dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice).head(rank_number) for df in collections]  ## 'dataframes' list 中的各dataframe已經是按照次數高至低的項目順序排列
        else:
            # dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice, row_rank, row_rank_number) for df in collections]
            dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice) for df in collections]

        
        ## 形成所有學系'項目'欄位的所有值
        # desired_order  = list(set([item for df in dataframes for item in df['項目'].tolist()])) 
        desired_order  = list(dict.fromkeys([item for df in dataframes for item in df['項目'].tolist()]))
        desired_order = desired_order[::-1]  ## 反轉次序使得表與圖的項目次序一致
        ## 缺的項目值加以擴充， 並統一一樣的項目次序
        dataframes = [adjust_df(df, desired_order) for df in dataframes]        
        combined_df = pd.concat(dataframes, keys=selected_options)
    elif 系_院_校 == '2':
        # collections = [df_junior_original[df_junior_original['學院'].str.contains(i, regex=True)] for i in selected_options if i!='全校' else df_junior_original]
        # collections = [df_junior_original] + collections
        collections = [df_junior_school if i == '全校' else df_junior_school[df_junior_school['學院']==i] for i in selected_options]

        
        if rank == True:
            # dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice, row_rank, row_rank_number).head(rank_number) for df in collections]  ## 'dataframes' list 中的各dataframe已經是按照次數高至低的項目順序排列
            dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice).head(rank_number) for df in collections]  ## 'dataframes' list 中的各dataframe已經是按照次數高至低的項目順序排列
        else:
            # dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice, row_rank, row_rank_number) for df in collections]
            dataframes = [Frequency_Distribution(df, column_index, split_symbol, dropped_string, sum_choice) for df in collections]
    
            
        ## 形成所有學系'項目'欄位的所有值
        # desired_order  = list(set([item for df in dataframes for item in df['項目'].tolist()])) 
        desired_order  = list(dict.fromkeys([item for df in dataframes for item in df['項目'].tolist()]))
        desired_order = desired_order[::-1]  ## 反轉次序使得表與圖的項目次序一致
        ## 缺的項目值加以擴充， 並統一一樣的項目次序
        dataframes = [adjust_df(df, desired_order) for df in dataframes]        
        combined_df = pd.concat(dataframes, keys=selected_options)
        # combined_df = pd.concat(dataframes, keys=['全校'])

            
    # 获取level 0索引的唯一值并保持原始顺序
    unique_level0 = combined_df.index.get_level_values(0).unique()

    #### 設置 matplotlib 支持中文的字體: 
    # matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    # matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
    matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # #### 设置条形的宽度
    # bar_width = 0.2
    #### 设置y轴的位置
    # r = np.arange(len(dataframes[0]))  ## len(result_df_理學_rr)=6, 因為result_df_理學_rr 有 6個 row: 非常滿意, 滿意, 普通, 不滿意, 非常不滿意
    r = np.arange(len(desired_order))
    # #### 设置字体大小
    # title_fontsize = 15
    # xlabel_fontsize = 14
    # ylabel_fontsize = 14
    # xticklabel_fontsize = 14
    # yticklabel_fontsize = 14
    # annotation_fontsize = 8
    # legend_fontsize = 14
    

    #### 绘制条形
    fig, ax = plt.subplots(figsize=(width3, heigh3))
    # if 系_院_校 == '0' or '1':
    # for i, (college_name, df) in enumerate(combined_df.groupby(level=0)):
    for i, college_name in enumerate(unique_level0):            
        df = combined_df.loc[college_name]
        # 计算当前分组的条形数量
        num_bars = len(df)
        # 生成当前分组的y轴位置
        index = np.arange(num_bars) + i * bar_width
        # index = r + i * bar_width
        # if 系_院_校 == '0' or '1':
        rects = ax.barh(index, df['比例'], height=bar_width, label=college_name)
    # if 系_院_校 == '2':
    # #     index = np.arange(len(desired_order))
    # #     rects = ax.barh(index, dataframes[0]['比例'], height=bar_width, label='全校')
    #     for i, college_name in enumerate(unique_level0):            
    #         df = combined_df.loc[college_name]
    #         # 计算当前分组的条形数量
    #         num_bars = len(df)
    #         # 生成当前分组的y轴位置
    #         index = np.arange(num_bars) + i * bar_width
    #         # index = r + i * bar_width
    #         # if 系_院_校 == '0' or '1':
    #         # rects = ax.barh(index, df['比例'], height=bar_width, label='全校')
    #         # if i==0:
    #         rects = ax.barh(index, df['比例'], height=bar_width, label=college_name)
    

        # # 在每个条形上标示比例
        # for rect, ratio in zip(rects, df['比例']):
        #     ax.text(rect.get_x() + rect.get_width() / 2.0, rect.get_height(), f'{ratio:.1%}', ha='center', va='bottom',fontsize=annotation_fontsize)
    
    # if 系_院_校 == '0' or '1':
    ### 添加图例
    if fontsize_adjust==0:
        ax.legend()
    if fontsize_adjust==1:
        ax.legend(fontsize=legend_fontsize)
    

    # ### 添加x轴标签
    # ## 计算每个组的中心位置作为x轴刻度位置
    # # group_centers = r + bar_width * (num_colleges / 2 - 0.5)
    # # group_centers = np.arange(len(dataframes[0]))
    # ## 添加x轴标签
    # # ax.set_xticks(group_centers)
    # # dataframes[0]['項目'].values
    # # "array(['個人興趣', '未來能找到好工作', '落點分析', '沒有特定理由', '家人的期望與建議', '師長推薦'],dtype=object)"
    # ax.set_xticks(r + bar_width * (len(dataframes) / 2))
    # ax.set_xticklabels(dataframes[0]['項目'].values, fontsize=xticklabel_fontsize)
    # # ax.set_xticklabels(['非常滿意', '滿意', '普通', '不滿意','非常不滿意'],fontsize=xticklabel_fontsize)

    ### 设置x,y轴刻度标签
    ax.set_yticks(r + bar_width*(len(dataframes) / 2))  # 调整位置以使标签居中对齐到每个条形
    if fontsize_adjust==0:
        # ax.set_yticklabels(dataframes[0]['項目'].values) 
        ax.set_yticklabels(desired_order)
        ax.tick_params(axis='x')
    if fontsize_adjust==1:
        # ax.set_yticklabels(dataframes[0]['項目'].values, fontsize=yticklabel_fontsize)
        ax.set_yticklabels(desired_order, fontsize=yticklabel_fontsize)
        ## 设置x轴刻度的字体大小
        ax.tick_params(axis='x', labelsize=xticklabel_fontsize)
        
    


    ### 设置标题和轴标签
    if fontsize_adjust==0:
        ax.set_title(item_name)
        ax.set_xlabel('比例')
    if fontsize_adjust==1:
        ax.set_title(item_name,fontsize=title_fontsize)
        ax.set_xlabel('比例',fontsize=xlabel_fontsize)
    
    
    
    ### 显示网格线
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    # plt.show()
    ### 在Streamlit中显示
    # if 系_院_校 == '0' or '1':
    st.pyplot(plt)









####### 設定呈現標題 
html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;"> 112學年度大三學生學習經驗調查 </h1>
		</div>
		"""
stc.html(html_temp)
# st.subheader("以下調查與計算母體為大二填答同學1834人")
###### 使用 <h3> 或 <h4> 标签代替更大的标题标签
# st.markdown("##### 以下調查與計算母體為大二填答同學1834人")

###### 或者，使用 HTML 的 <style> 来更精细地控制字体大小和加粗
st.markdown("""
<style>
.bold-small-font {
    font-size:18px !important;
    font-weight:bold !important;
}
</style>
<p class="bold-small-font">以下調查與計算母體為大三填答同學 2189人</p>
""", unsafe_allow_html=True)

st.markdown("##")  ## 更大的间隔


university_faculties_list = ['全校','理學院','資訊學院','管理學院','人社院','外語學院','國際學院']
# global 系_院_校
####### 選擇系院校
###### 選擇 系 or 院 or 校:
系_院_校 = st.text_input('以學系查詢請輸入 0, 以學院查詢請輸入 1, 以全校查詢請輸入 2  (說明: (i).以學系查詢時同時呈現學院及全校資料. (ii)可以選擇比較單位): ', value='0')
if 系_院_校 == '0':
    choice = st.selectbox('選擇學系', df_junior_original['科系'].unique(), index=0)
    choice = '大傳系'
    df_junior = df_junior_original[df_junior_original['科系']==choice]
    choice_faculty = df_junior['學院'].values[0]  ## 選擇學系所屬學院
    df_junior_faculty = df_junior_original[df_junior_original['學院']==choice_faculty]  ## 挑出全校所屬學院之資料

    # selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=['化科系','企管系'])
    # selected_options = ['化科系','企管系']
    # collections = [df_junior_original[df_junior_original['科系']==i] for i in selected_options]
    # dataframes = [Frequency_Distribution(df, 7) for df in collections]
    # combined_df = pd.concat(dataframes, keys=selected_options)
    # #### 去掉 level 1 index
    # combined_df_r = combined_df.reset_index(level=1, drop=True)
elif 系_院_校 == '1':
    choice = st.selectbox('選擇學院', df_junior_original['學院'].unique(),index=0)
    #choice = '管理'
    df_junior = df_junior_original[df_junior_original['學院']==choice]
    # selected_options = st.multiselect('選擇比較學的院：', df_junior_original['學院'].unique(), default=['理學院','資訊學院'])
    # collections = [df_junior_original[df_junior_original['學院']==i] for i in selected_options]
    # dataframes = [Frequency_Distribution(df, 7) for df in collections]
    # combined_df = pd.concat(dataframes, keys=selected_options)
    df_junior_faculty = df_junior   ## 沒有用途, 只是為了不要讓 Draw() 中的參數 'df_junior_faculty' 缺漏
elif 系_院_校 == '2':
    choice = '全校'
    # choice = st.selectbox('選擇:全校', university_list, index=0)
    # if choice !='全校':
    #     df_junior = df_junior_original[df_junior_original['學院'].str.contains(choice, regex=True)]
    # if choice !='全校':
    #     df_junior = df_junior_original
    
    df_junior = df_junior_original  ## 
    df_junior_faculty = df_junior  ## 沒有用途, 只是為了不要讓 Draw() 中的參數 'df_junior_faculty' 缺漏




# choice = st.selectbox('選擇學系', df_junior_original['科系'].unique())
# #choice = '化科系'
# df_junior = df_junior_original[df_junior_original['科系']==choice]
# selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique())
# # selected_options = ['化科系','企管系']
# collections = [df_junior_original[df_junior_original['科系']==i] for i in selected_options]
# dataframes = [Frequency_Distribution(df, 7) for df in collections]
# combined_df = pd.concat(dataframes, keys=selected_options)
# # combined_df = pd.concat([dataframes[0], dataframes[1]], axis=0)



df_streamlit = []
column_title = []


####### 問卷的各項問題
st.markdown("""
<style>
.bold-small-font {
    font-size:18px !important;
    font-weight:bold !important;
}
</style>
<p class="bold-small-font">基本資料</p>
""", unsafe_allow_html=True)

###### 1-1.各班級填答人數與填答比例
with st.expander("1-1.各班級填答人數與填答比例:"):
    
    item_name = "各班級填答人數與填答比例"
    #### 各班人數:
    df_ID_departments_unique_counts = df_ID.groupby('班級')['email'].nunique()
    # print(df_ID_departments_unique_counts)
    # '''
    # 班級
    # 中三A             49
    # 中三B             44
    # 企管三A            48
    # 企管三B            47
    # 企管三C            48
    # 化科三A            50
    # 化科三B            50
    # 化科三C            16
    # 台文三A            48
    # 國企三A            54
    # 國企三B            53
    # 國企三C            53
    # 國際資訊學士學位學程三A    10
    # 大傳三A            64
    # 寰宇外語教育三A        26
    # 寰宇管理學程三A        48
    # 應化三A            46
    # 應化三B            50
    # 日三A             66
    # 日三B             66
    # 會計三A            49
    # 會計三B            47
    # 會計三C            51
    # 法律三A            56
    # 法律三B            58
    # 法律原專三A           5
    # 生態三A            51
    # 社工三A            60
    # 社工三B            63
    # 社工三C             6
    # 社工原專三A          12
    # 英三A             49
    # 英三B             49
    # 西三A             41
    # 西三B             42
    # 觀光三A            54
    # 觀光三B            58
    # 觀光三C            54
    # 財工三A            45
    # 財金三A            53
    # 財金三B            56
    # 財金三C            57
    # 資傳三A            55
    # 資傳三B            60
    # 資工三A            61
    # 資工三B            62
    # 資科三A            46
    # 資科三B            40
    # 資管三A            53
    # 資管三B            63
    # 食營三-營養組         55
    # 食營三-食品組         44
    # Name: email, dtype: int64
    # '''
    #### 填答人數
    df_junior_original_departments_unique_counts = df_junior_original.groupby('班級')['email'].nunique()
    # type(df_junior_original_departments_unique_counts)  ## pandas.core.series.Series
    # len(df_junior_original_departments_unique_counts)  ## 51, 與 df_ID_departments_unique_counts 對照少了 "國際資訊學士學位學程三A"
    # print(df_junior_original_departments_unique_counts)
    # '''
    # 班級
    # 中三A         41
    # 中三B         40
    # 企管三A        34
    # 企管三B        41
    # 企管三C        45
    # 化科三A        46
    # 化科三B        49
    # 化科三C        13
    # 台文三A        47
    # 國企三A        50
    # 國企三B        51
    # 國企三C        45
    # 大傳三A        40
    # 寰宇外語教育三A    24
    # 寰宇管理學程三A    34
    # 應化三A        42
    # 應化三B        45
    # 日三A         59
    # 日三B         54
    # 會計三A        45
    # 會計三B        43
    # 會計三C        47
    # 法律三A        47
    # 法律三B        42
    # 法律原專三A       4
    # 生態三A        44
    # 社工三A        54
    # 社工三B        62
    # 社工三C         6
    # 社工原專三A      12
    # 英三A         45
    # 英三B         44
    # 西三A         34
    # 西三B         41
    # 觀光三A        47
    # 觀光三B        49
    # 觀光三C        52
    # 財工三A        41
    # 財金三A        45
    # 財金三B        46
    # 財金三C        52
    # 資傳三A        47
    # 資傳三B        56
    # 資工三A        58
    # 資工三B        49
    # 資科三A        42
    # 資科三B        37
    # 資管三A        48
    # 資管三B        56
    # 食營三-營養組     55
    # 食營三-食品組     39
    # Name: email, dtype: int64
    # '''

    ### 添加新的索引 "國際資訊學士學位學程三A" 並設置值為 0
    df_junior_original_departments_unique_counts['國際資訊學士學位學程三A'] = 0
    # print(df_junior_original_departments_unique_counts)
    # '''
    # 班級
    # 中三A             41
    # 中三B             40
    # 企管三A            34
    # 企管三B            41
    # 企管三C            45
    # 化科三A            46
    # 化科三B            49
    # 化科三C            13
    # 台文三A            47
    # 國企三A            50
    # 國企三B            51
    # 國企三C            45
    # 大傳三A            40
    # 寰宇外語教育三A        24
    # 寰宇管理學程三A        34
    # 應化三A            42
    # 應化三B            45
    # 日三A             59
    # 日三B             54
    # 會計三A            45
    # 會計三B            43
    # 會計三C            47
    # 法律三A            47
    # 法律三B            42
    # 法律原專三A           4
    # 生態三A            44
    # 社工三A            54
    # 社工三B            62
    # 社工三C             6
    # 社工原專三A          12
    # 英三A             45
    # 英三B             44
    # 西三A             34
    # 西三B             41
    # 觀光三A            47
    # 觀光三B            49
    # 觀光三C            52
    # 財工三A            41
    # 財金三A            45
    # 財金三B            46
    # 財金三C            52
    # 資傳三A            47
    # 資傳三B            56
    # 資工三A            58
    # 資工三B            49
    # 資科三A            42
    # 資科三B            37
    # 資管三A            48
    # 資管三B            56
    # 食營三-營養組         55
    # 食營三-食品組         39
    # 國際資訊學士學位學程三A     0
    # Name: email, dtype: int64
    # '''

    #### 填答比例
    ### 合并为DataFrame
    df_填答比例 = pd.concat([df_ID_departments_unique_counts, df_junior_original_departments_unique_counts], axis=1)
    # type(df_填答比例)  ## pandas.core.frame.DataFrame
    ### 修改欄位名稱
    df_填答比例.columns = ['學生人數','填答人數']
    ### 计算两行的比例并创建新行
    df_填答比例['填答比例'] = df_填答比例['填答人數'] / df_填答比例['學生人數']
    # len(df_填答比例)  ## 52
    # print(df_填答比例['填答比例'])
    # '''
    # 班級
    # 中三A             0.836735
    # 中三B             0.909091
    # 企管三A            0.708333
    # 企管三B            0.872340
    # 企管三C            0.937500
    # 化科三A            0.920000
    # 化科三B            0.980000
    # 化科三C            0.812500
    # 台文三A            0.979167
    # 國企三A            0.925926
    # 國企三B            0.962264
    # 國企三C            0.849057
    # 國際資訊學士學位學程三A    0.000000
    # 大傳三A            0.625000
    # 寰宇外語教育三A        0.923077
    # 寰宇管理學程三A        0.708333
    # 應化三A            0.913043
    # 應化三B            0.900000
    # 日三A             0.893939
    # 日三B             0.818182
    # 會計三A            0.918367
    # 會計三B            0.914894
    # 會計三C            0.921569
    # 法律三A            0.839286
    # 法律三B            0.724138
    # 法律原專三A          0.800000
    # 生態三A            0.862745
    # 社工三A            0.900000
    # 社工三B            0.984127
    # 社工三C            1.000000
    # 社工原專三A          1.000000
    # 英三A             0.918367
    # 英三B             0.897959
    # 西三A             0.829268
    # 西三B             0.976190
    # 觀光三A            0.870370
    # 觀光三B            0.844828
    # 觀光三C            0.962963
    # 財工三A            0.911111
    # 財金三A            0.849057
    # 財金三B            0.821429
    # 財金三C            0.912281
    # 資傳三A            0.854545
    # 資傳三B            0.933333
    # 資工三A            0.950820
    # 資工三B            0.790323
    # 資科三A            0.913043
    # 資科三B            0.925000
    # 資管三A            0.905660
    # 資管三B            0.888889
    # 食營三-營養組         1.000000
    # 食營三-食品組         0.886364
    # Name: 填答比例, dtype: float64
    # '''

    ### 使用 reset_index 方法將索引變為列
    df_填答比例 = df_填答比例.reset_index()
    
    
    
    
    
    


    # ##### 產出 result_df
    # result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    # ##### 存到 list 'df_streamlit'
    # df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame 
    # st.write(choice)
    st.write(f"<h6>{item_name}</h6>", unsafe_allow_html=True)
    st.write(df_填答比例.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔
    
    
    
###### 1-2.身分別.
with st.expander("1-2.身分別:"):
    
    # item_name = "身分別"
    # df_身分別 = df_junior_original['身分'].value_counts(ascending=False)
    # # '''
    # # 本地生    2098
    # # 外籍生      70
    # # 僑生       21
    # # Name: 身分, dtype: int64
    # # '''
    
    # ##### 使用 reset_index 方法將 Series 'df_身分別' 轉換為 DataFrame
    # df_身分別_df = df_身分別.reset_index()
    
    # # 重命名新的 DataFrame 的欄位
    # df_身分別_df.columns = ['身分別', '人數']
    
    # ##### 使用Streamlit展示DataFrame 
    # # st.write(choice)
    # st.write(f"<h6>{item_name}</h6>", unsafe_allow_html=True)
    # st.write(df_身分別_df.to_html(index=False), unsafe_allow_html=True)
    # st.markdown("##")  ## 更大的间隔
    
    column_index = 66
    item_name = "身分別"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)
st.markdown("##")  ## 更大的间隔 
    
    
    
st.markdown("""
<style>
.bold-small-font {
    font-size:18px !important;
    font-weight:bold !important;
}
</style>
<p class="bold-small-font">工讀情況</p>
""", unsafe_allow_html=True)

###### 2-1.您三年級就學期間是否曾工讀？
with st.expander("2-1.您三年級就學期間是否曾工讀:"):
    # df_junior.iloc[:,9] ## 
    column_index = 9
    item_name = "三年級就學期間是否曾工讀"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 2-2.您三年級「上學期」平均每周工讀時數？
with st.expander("2-2.三年級「上學期」平均每周工讀時數(不列計沒有工讀):"):
    # df_junior.iloc[:,10] ##   df_junior.iloc[:,9].unique() ## array(['是', '否'], dtype=object)
    column_index = 10
    item_name = "三年級「上學期」平均每周工讀時數(不列計沒有工讀)"
    column_title.append(df_junior.columns[column_index][0:])
    



    # ##### 加條件: 2-1 回答 '是' 者, 才能進行此題2-2. 使用布林索引過濾掉 第9行中包含 '否' 的 rows, 因為大三完全沒有工讀的不列入考慮
    # if 系_院_校 == '0':
    #     df_junior_restrict = df_junior[~df_junior.iloc[:, 9].str.contains('否')]
    #     df_junior_faculty_restrict = df_junior_faculty[~df_junior_faculty.iloc[:, 9].str.contains('否')]
    #     df_junior_school_restrict = df_junior_original[~df_junior_original.iloc[:, 9].str.contains('否')]
        
    # if 系_院_校 == '1':
    #     df_junior_restrict = df_junior[~df_junior.iloc[:, 9].str.contains('否')]
    #     df_junior_faculty_restrict = df_junior_restrict  ## 沒有作用
    #     df_junior_school_restrict = df_junior_original[~df_junior_original.iloc[:, 9].str.contains('否')]
    # if 系_院_校 == '2':
    #     df_junior_restrict = df_junior[~df_junior.iloc[:, 9].str.contains('否')]
    #     df_junior_faculty_restrict = df_junior_restrict  ## 沒有作用
    #     df_junior_school_restrict = df_junior_restrict




    ##### 產出 result_df
    # result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0)
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔 



###### 2-3.您三年級「上學期」的工讀地點為何？
with st.expander("2-3.三年級「上學期」的工讀地點(不列計沒有工讀):"):
    # df_junior.iloc[:,11] ##   
    column_index = 11
    item_name = "三年級「上學期」的工讀地點(不列計沒有工讀)"
    column_title.append(df_junior.columns[column_index][0:])
    

    ##### 產出 result_df
    # result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0)
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔 



###### 2-4.您三年級「下學期」平均每周工讀時數？
with st.expander("2-4.三年級「下學期」平均每周工讀時數(不列計沒有工讀):"):
    # df_junior.iloc[:,12] ##  
    column_index = 12
    item_name = "三年級「下學期」平均每周工讀時數(不列計沒有工讀)"
    column_title.append(df_junior.columns[column_index][0:])
    


    ##### 產出 result_df
    # result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0)
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔 



###### 2-5.您三年級「下學期」的工讀地點為何？
with st.expander("2-5.三年級「下學期」的工讀地點(不列計沒有工讀):"):
    # df_junior.iloc[:,13] ##   
    column_index = 13
    item_name = "三年級「下學期」的工讀地點(不列計沒有工讀)"
    column_title.append(df_junior.columns[column_index][0:])
    

    ##### 產出 result_df
    # result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0)
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔 



###### 2-6-1.您工讀的原因為何？(直接拖拉，依原因之優先順序排列，最主要原因放在最上方): 第一順位 
with st.expander("2-6-1.工讀的原因:第一順位(不列計沒有工讀):"):
    # df_junior.iloc[:,14] ##   
    column_index = 14
    item_name = "工讀的原因:第一順位(不列計沒有工讀)"
    column_title.append(df_junior.columns[column_index][0:])
    # ##### 使用 str.split 方法分割第14行的字串，以 ';' 為分隔符, 然後使用 apply 和 lambda 函數來提取前三個元素, 並再度以;分隔.
    # # df_junior['col14'] = df_junior['col14'].str.split(';').apply(lambda x: ';'.join(x[:3]))
    # df_junior.iloc[:,column_index] = df_junior.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[:3]))


    
    ##### 產出 result_df: 加條件: 
    ranking_number = 1
    
    # if 系_院_校 == '0':
    #     df_junior_restrict = df_junior.dropna(subset=[df_junior.columns[column_index]])
    #     df_junior_restrict.iloc[:,column_index] = df_junior_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[:ranking_number]))
    #     df_junior_faculty_restrict = df_junior_faculty.dropna(subset=[df_junior_faculty.columns[column_index]])
    #     df_junior_faculty_restrict.iloc[:,column_index] = df_junior_faculty_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[:ranking_number]))
    #     df_junior_school_restrict = df_junior_original.dropna(subset=[df_junior_original.columns[column_index]])
    #     df_junior_school_restrict.iloc[:,column_index] = df_junior_school_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[:ranking_number]))
        
        
    # if 系_院_校 == '1':
    #     df_junior_restrict = df_junior.dropna(subset=[df_junior.columns[column_index]])
    #     df_junior_restrict.iloc[:,column_index] = df_junior_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[:ranking_number]))
    #     df_junior_faculty_restrict = df_junior_restrict  ## 沒有作用
    #     df_junior_school_restrict = df_junior_original.dropna(subset=[df_junior_original.columns[column_index]])
    #     df_junior_school_restrict.iloc[:,column_index] = df_junior_school_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[:ranking_number]))
    # if 系_院_校 == '2':
    #     df_junior_restrict = df_junior.dropna(subset=[df_junior.columns[column_index]])
    #     df_junior_restrict.iloc[:,column_index] = df_junior_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[:ranking_number]))
    #     df_junior_faculty_restrict = df_junior_restrict  ## 沒有作用
    #     df_junior_school_restrict = df_junior_restrict
        
        
    if 系_院_校 == '0':
        df_junior_restrict = df_junior.dropna(subset=[df_junior.columns[column_index]])
        df_junior_restrict.iloc[:,column_index] = df_junior_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        df_junior_faculty_restrict = df_junior_faculty.dropna(subset=[df_junior_faculty.columns[column_index]])
        df_junior_faculty_restrict.iloc[:,column_index] = df_junior_faculty_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        df_junior_school_restrict = df_junior_original.dropna(subset=[df_junior_original.columns[column_index]])
        df_junior_school_restrict.iloc[:,column_index] = df_junior_school_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        
        
    if 系_院_校 == '1':
        df_junior_restrict = df_junior.dropna(subset=[df_junior.columns[column_index]])
        df_junior_restrict.iloc[:,column_index] = df_junior_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        df_junior_faculty_restrict = df_junior_restrict  ## 沒有作用
        df_junior_school_restrict = df_junior_original.dropna(subset=[df_junior_original.columns[column_index]])
        df_junior_school_restrict.iloc[:,column_index] = df_junior_school_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
    if 系_院_校 == '2':
        df_junior_restrict = df_junior.dropna(subset=[df_junior.columns[column_index]])
        df_junior_restrict.iloc[:,column_index] = df_junior_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        df_junior_faculty_restrict = df_junior_restrict  ## 沒有作用
        df_junior_school_restrict = df_junior_restrict




    

    ##### 產出 result_df
    # result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0)
    # result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1 , row_rank=True, row_rank_number=3)
    result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d1')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f1')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university1')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order, row_rank=True, row_rank_number=1) 
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order)
st.markdown("##")  ## 更大的间隔 



###### 2-6-2.您工讀的原因為何？(直接拖拉，依原因之優先順序排列，最主要原因放在最上方): 第二順位 
with st.expander("2-6-2.工讀的原因:第二順位(不列計沒有工讀):"):
    # df_junior.iloc[:,14] ##   
    column_index = 14
    item_name = "工讀的原因:第二順位(不列計沒有工讀)"
    column_title.append(df_junior.columns[column_index][0:])
    # ##### 使用 str.split 方法分割第14行的字串，以 ';' 為分隔符, 然後使用 apply 和 lambda 函數來提取前三個元素, 並再度以;分隔.
    # # df_junior['col14'] = df_junior['col14'].str.split(';').apply(lambda x: ';'.join(x[:3]))
    # df_junior.iloc[:,column_index] = df_junior.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[:3]))


    
    ##### 產出 result_df: 加條件: 
    ranking_number = 2
        
    if 系_院_校 == '0':
        df_junior_restrict = df_junior.dropna(subset=[df_junior.columns[column_index]])
        df_junior_restrict.iloc[:,column_index] = df_junior_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        df_junior_faculty_restrict = df_junior_faculty.dropna(subset=[df_junior_faculty.columns[column_index]])
        df_junior_faculty_restrict.iloc[:,column_index] = df_junior_faculty_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        df_junior_school_restrict = df_junior_original.dropna(subset=[df_junior_original.columns[column_index]])
        df_junior_school_restrict.iloc[:,column_index] = df_junior_school_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
    if 系_院_校 == '1':
        df_junior_restrict = df_junior.dropna(subset=[df_junior.columns[column_index]])
        df_junior_restrict.iloc[:,column_index] = df_junior_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        df_junior_faculty_restrict = df_junior_restrict  ## 沒有作用
        df_junior_school_restrict = df_junior_original.dropna(subset=[df_junior_original.columns[column_index]])
        df_junior_school_restrict.iloc[:,column_index] = df_junior_school_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
    if 系_院_校 == '2':
        df_junior_restrict = df_junior.dropna(subset=[df_junior.columns[column_index]])
        df_junior_restrict.iloc[:,column_index] = df_junior_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        df_junior_faculty_restrict = df_junior_restrict  ## 沒有作用
        df_junior_school_restrict = df_junior_restrict




    

    ##### 產出 result_df
    # result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0)
    # result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1 , row_rank=True, row_rank_number=3)
    result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d2')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f2')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university2')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order, row_rank=True, row_rank_number=1)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 2-6-3.您工讀的原因為何？(直接拖拉，依原因之優先順序排列，最主要原因放在最上方): 第三順位 
with st.expander("2-6-3.工讀的原因:第三順位(不列計沒有工讀):"):
    # df_junior.iloc[:,14] ##   
    column_index = 14
    item_name = "工讀的原因:第三順位(不列計沒有工讀)"
    column_title.append(df_junior.columns[column_index][0:])
    # ##### 使用 str.split 方法分割第14行的字串，以 ';' 為分隔符, 然後使用 apply 和 lambda 函數來提取前三個元素, 並再度以;分隔.
    # # df_junior['col14'] = df_junior['col14'].str.split(';').apply(lambda x: ';'.join(x[:3]))
    # df_junior.iloc[:,column_index] = df_junior.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[:3]))


    
    ##### 產出 result_df: 加條件: 
    ranking_number = 3
    
    if 系_院_校 == '0':
        df_junior_restrict = df_junior.dropna(subset=[df_junior.columns[column_index]])
        df_junior_restrict.iloc[:,column_index] = df_junior_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        df_junior_faculty_restrict = df_junior_faculty.dropna(subset=[df_junior_faculty.columns[column_index]])
        df_junior_faculty_restrict.iloc[:,column_index] = df_junior_faculty_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        df_junior_school_restrict = df_junior_original.dropna(subset=[df_junior_original.columns[column_index]])
        df_junior_school_restrict.iloc[:,column_index] = df_junior_school_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
    if 系_院_校 == '1':
        df_junior_restrict = df_junior.dropna(subset=[df_junior.columns[column_index]])
        df_junior_restrict.iloc[:,column_index] = df_junior_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        df_junior_faculty_restrict = df_junior_restrict  ## 沒有作用
        df_junior_school_restrict = df_junior_original.dropna(subset=[df_junior_original.columns[column_index]])
        df_junior_school_restrict.iloc[:,column_index] = df_junior_school_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
    if 系_院_校 == '2':
        df_junior_restrict = df_junior.dropna(subset=[df_junior.columns[column_index]])
        df_junior_restrict.iloc[:,column_index] = df_junior_restrict.iloc[:,column_index].str.split(';').apply(lambda x: ';'.join(x[ranking_number-1:ranking_number]))
        df_junior_faculty_restrict = df_junior_restrict  ## 沒有作用
        df_junior_school_restrict = df_junior_restrict




    

    ##### 產出 result_df
    # result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0)
    # result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1 , row_rank=True, row_rank_number=3)
    result_df = Frequency_Distribution(df_junior_restrict, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d3')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f3')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university3')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=0, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order, row_rank=True, row_rank_number=1)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order)
st.markdown("##")  ## 更大的间隔  



st.markdown("""
<style>
.bold-small-font {
    font-size:18px !important;
    font-weight:bold !important;
}
</style>
<p class="bold-small-font">課外活動參與情況</p>
""", unsafe_allow_html=True)

###### 3-1.您認為下列哪些經驗對未來工作會有所幫助？(可複選)
with st.expander("3-1.哪些經驗對未來工作會有所幫助(複選):"):
    # df_junior.iloc[:,16] ## 
    column_index = 16
    item_name = "哪些經驗對未來工作會有所幫助(複選)"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior_restrict, df_junior_faculty=df_junior_faculty_restrict, df_junior_school=df_junior_school_restrict, desired_order=desired_order) 
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)
st.markdown("##")  ## 更大的间隔



###### 3-2.您參與過哪些課外活動: 參加社團
with st.expander("3-2.是否參加社團:"):
    # df_junior.iloc[:,17] ## 
    column_index = 17
    item_name = "是否參加社團"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 3-3.您參與過哪些課外活動: 擔任社團/系學會幹部
with st.expander("3-3.是否擔任社團/系學會幹部:"):
    # df_junior.iloc[:,18] ## 
    column_index = 18
    item_name = "是否擔任社團/系學會幹部"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 3-4.您參與過哪些課外活動: 參加志工服務
with st.expander("3-4.是否參加志工服務:"):
    # df_junior.iloc[:,19] ## 
    column_index = 19
    item_name = "是否參加志工服務"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 3-5.您參與過哪些課外活動: 參加校外實習
with st.expander("3-5.是否參加校外實習:"):
    # df_junior.iloc[:,20] ## 
    column_index = 20
    item_name = "是否參加校外實習"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 3-6.您參與過哪些課外活動: 參加企業參訪
with st.expander("3-6.是否參加企業參訪:"):
    # df_junior.iloc[:,21] ## 
    column_index = 21
    item_name = "是否參加企業參訪"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔





###### 3-7.上述您參與過的活動中，您認為哪些經驗對未來工作會有所幫助？(可複選)
with st.expander("3-7.上述參與過的活動中，哪些經驗對未來工作會有所幫助(可複選):"):
    # df_junior.iloc[:,23] ## 
    column_index = 23
    item_name = "上述參與過的活動中，哪些經驗對未來工作會有所幫助(可複選)"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
        

    # Draw(系_院_校, column_index, ';', '沒有工讀', 1, result_df, selected_options, dataframes, combined_df, bar_width = 0.15)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df, selected_options)
    # Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14,bar_width = 0.2, fontsize_adjust=0)
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



st.markdown("""
<style>
.bold-small-font {
    font-size:18px !important;
    font-weight:bold !important;
}
</style>
<p class="bold-small-font">學習情況與能力培養</p>
""", unsafe_allow_html=True)


###### 4-1.您在專業課程學習上的投入/認真程度？ (依多數課程情況回答)
with st.expander("4-1.在專業課程學習上的投入/認真程度:"):
    # df_junior.iloc[:,25] ## 
    column_index = 25
    item_name = "在專業課程學習上的投入/認真程度"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=False, rank_number=5, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 4-2.請對專業課程提供意見或建議.
with st.expander("4-2.對專業課程提供意見或建議:"):
    # df_junior.iloc[:,26] ## 
    column_index = 26
    item_name = "對專業課程提供意見或建議(部分呈現)"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 4-3.您認為下列哪些能力有助於提升就業力？(可複選)
with st.expander("4-3.哪些能力有助於提升就業力:"):
    # df_junior.iloc[:,27] ## 
    column_index = 27
    item_name = "哪些能力有助於提升就業力"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 4-4.與大一入學時相比較, 您目前在下列各項能力是否有提升？ 專業能力
with st.expander("4-4.與大一入學時相比較「專業能力」是否提升"):
    # df_junior.iloc[:,28] ## 
    column_index = 28
    item_name = "與大一入學時相比較「專業能力」是否提升"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 4-5.與大一入學時相比較, 您目前在下列各項能力是否有提升？ 溝通表達能力
with st.expander("4-5.與大一入學時相比較「溝通表達能力」是否提升"):
    # df_junior.iloc[:,29] ## 
    column_index = 29
    item_name = "與大一入學時相比較「溝通表達能力」是否提升"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 4-6.與大一入學時相比較, 您目前在下列各項能力是否有提升？ 團隊合作
with st.expander("4-6.與大一入學時相比較「團隊合作」是否提升"):
    # df_junior.iloc[:,30] ## 
    column_index = 30
    item_name = "與大一入學時相比較「團隊合作」是否提升"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 4-7.與大一入學時相比較, 您目前在下列各項能力是否有提升？ 問題解決能力
with st.expander("4-7.與大一入學時相比較「問題解決能力」是否提升"):
    # df_junior.iloc[:,31] ## 
    column_index = 31
    item_name = "與大一入學時相比較「問題解決能力」是否提升"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 4-8.與大一入學時相比較, 您目前在下列各項能力是否有提升？ 自主探索學習能力
with st.expander("4-8.與大一入學時相比較「自主探索學習能力」是否提升"):
    # df_junior.iloc[:,32] ## 
    column_index = 32
    item_name = "與大一入學時相比較「自主探索學習能力」是否提升"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 4-9.與大一入學時相比較, 您目前在下列各項能力是否有提升？ 分析思考能力
with st.expander("4-9.與大一入學時相比較「分析思考能力」是否提升"):
    # df_junior.iloc[:,33] ## 
    column_index = 33
    item_name = "與大一入學時相比較「分析思考能力」是否提升"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 4-10.與大一入學時相比較, 您目前在下列各項能力是否有提升？ 領導能力
with st.expander("4-10.與大一入學時相比較「領導能力」是否提升"):
    # df_junior.iloc[:,34] ## 
    column_index = 34
    item_name = "與大一入學時相比較「領導能力」是否提升"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 4-11.與大一入學時相比較, 您目前在下列各項能力是否有提升？ 創新能力
with st.expander("4-11.與大一入學時相比較「創新能力」是否提升"):
    # df_junior.iloc[:,35] ## 
    column_index = 35
    item_name = "與大一入學時相比較「創新能力」是否提升"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 4-12.與大一入學時相比較, 您目前在下列各項能力是否有提升？ 人際互動
with st.expander("4-12.與大一入學時相比較「人際互動」是否提升"):
    # df_junior.iloc[:,36] ## 
    column_index = 36
    item_name = "與大一入學時相比較「人際互動」是否提升"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 4-13.與大一入學時相比較, 您目前在下列各項能力是否有提升？ 外語能力
with st.expander("4-13.與大一入學時相比較「外語能力」是否提升"):
    # df_junior.iloc[:,37] ## 
    column_index = 37
    item_name = "與大一入學時相比較「外語能力」是否提升"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



st.markdown("""
<style>
.bold-small-font {
    font-size:18px !important;
    font-weight:bold !important;
}
</style>
<p class="bold-small-font">大四或畢業後之規劃</p>
""", unsafe_allow_html=True)

###### 5-1.大四或畢業後規劃參加項目「專業證照考試」
with st.expander("5-1.大四或畢業後規劃參加項目「專業證照考試」"):
    # df_junior.iloc[:,39] ## 
    column_index = 39
    item_name = "大四或畢業後規劃參加項目「專業證照考試」"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 5-2.大四或畢業後規劃參加項目「校外實習」
with st.expander("5-2.大四或畢業後規劃參加項目「校外實習」"):
    # df_junior.iloc[:,40] ## 
    column_index = 40
    item_name = "大四或畢業後規劃參加項目「校外實習」"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 5-3.大四或畢業後規劃參加項目「申請(考)研究所」
with st.expander("5-3.大四或畢業後規劃參加項目「申請(考)研究所」"):
    # df_junior.iloc[:,41] ## 
    column_index = 41
    item_name = "大四或畢業後規劃參加項目「申請(考)研究所」"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 5-4.大四或畢業後規劃參加項目「外語能力鑑定考試」
with st.expander("5-4.大四或畢業後規劃參加項目「外語能力鑑定考試」"):
    # df_junior.iloc[:,42] ## 
    column_index = 42
    item_name = "大四或畢業後規劃參加項目「外語能力鑑定考試」"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 5-5.大四或畢業後規劃參加項目「公職考試」
with st.expander("5-5.大四或畢業後規劃參加項目「公職考試」"):
    # df_junior.iloc[:,43] ## 
    column_index = 43
    item_name = "大四或畢業後規劃參加項目「公職考試」"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 5-6.大四或畢業後規劃參加項目「遊留學」
with st.expander("5-6.大四或畢業後規劃參加項目「遊留學」"):
    # df_junior.iloc[:,44] ## 
    column_index = 44
    item_name = "大四或畢業後規劃參加項目「遊留學」"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



st.markdown("""
<style>
.bold-small-font {
    font-size:18px !important;
    font-weight:bold !important;
}
</style>
<p class="bold-small-font">學生學習資源與輔導資源滿意度</p>
""", unsafe_allow_html=True)


###### 6-1.學生學習資源與輔導資源滿意度「學生學習輔導方案(學習輔導/自主學習/飛鷹助學)」
with st.expander("6-1.「學生學習輔導方案(學習輔導/自主學習/飛鷹助學)」滿意度"):
    # df_junior.iloc[:,46] ## 
    column_index = 46
    item_name = "「學生學習輔導方案(學習輔導/自主學習/飛鷹助學)」滿意度"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 6-2.學生學習資源與輔導資源滿意度「生活相關輔導(導師/領頭羊)」
with st.expander("6-2.「生活相關輔導(導師/領頭羊)」滿意度"):
    # df_junior.iloc[:,47] ## 
    column_index = 47
    item_name = "「生活相關輔導(導師/領頭羊)」滿意度"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 6-3.學生學習資源與輔導資源滿意度「職涯輔導」
with st.expander("6-3.「職涯輔導」滿意度"):
    # df_junior.iloc[:,48] ## 
    column_index = 48
    item_name = "「職涯輔導」滿意度"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 6-4.學生學習資源與輔導資源滿意度「外語教學中心學習輔導」
with st.expander("6-4.「外語教學中心學習輔導」滿意度"):
    # df_junior.iloc[:,49] ## 
    column_index = 49
    item_name = "「外語教學中心學習輔導」滿意度"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 6-5.學生學習資源與輔導資源滿意度「諮商暨健康中心的諮商輔導」
with st.expander("6-5.「諮商暨健康中心的諮商輔導」滿意度"):
    # df_junior.iloc[:,50] ## 
    column_index = 50
    item_name = "「諮商暨健康中心的諮商輔導」滿意度"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



###### 6-6.學生學習資源與輔導資源滿意度「國際化資源」
with st.expander("6-6.「國際化資源」滿意度"):
    # df_junior.iloc[:,51] ## 
    column_index = 51
    item_name = "「國際化資源」滿意度"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔



st.markdown("""
<style>
.bold-small-font {
    font-size:18px !important;
    font-weight:bold !important;
}
</style>
<p class="bold-small-font">生涯就業輔導</p>
""", unsafe_allow_html=True)


###### 7-1.生涯就業輔導: 學校能有效輔導同學就業與生涯發展規劃
with st.expander("7-1.學校能有效輔導同學就業與生涯發展規劃"):
    # df_junior.iloc[:,53] ## 
    column_index = 53
    item_name = "學校能有效輔導同學就業與生涯發展規劃"
    column_title.append(df_junior.columns[column_index][0:])


    ##### 產出 result_df
    result_df = Frequency_Distribution(df_junior, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1)

    ##### 存到 list 'df_streamlit'
    df_streamlit.append(result_df)  

    ##### 使用Streamlit展示DataFrame "result_df"，但不显示索引
    # st.write(choice)
    st.write(f"<h6>{choice}</h6>", unsafe_allow_html=True)
    st.write(result_df.to_html(index=False), unsafe_allow_html=True)
    st.markdown("##")  ## 更大的间隔

    ##### 使用Streamlit畫單一圖 & 比較圖
    #### 畫比較圖時, 比較單位之選擇:
    if 系_院_校 == '0':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學系：', df_junior_original['科系'].unique(), default=[choice,'企管系'],key=str(column_index)+'d')  ## # selected_options = ['化科系','企管系']
    if 系_院_校 == '1':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('選擇比較學院：', df_junior_original['學院'].unique(), default=[choice,'資訊學院'],key=str(column_index)+'f')
    if 系_院_校 == '2':
        ## 使用multiselect组件让用户进行多重选择
        selected_options = st.multiselect('比較選擇: 全校 or 各院：', university_faculties_list, default=['全校','理學院'],key=str(column_index)+'university')
    Draw(系_院_校, column_index, split_symbol=';', dropped_string='沒有工讀', sum_choice=1, result_df=result_df, selected_options=selected_options, dataframes=dataframes, combined_df=combined_df, width1=10,heigh1=6,width2=11,heigh2=8,width3=10,heigh3=6,title_fontsize=15,xlabel_fontsize = 14,ylabel_fontsize = 14,legend_fontsize = 14,xticklabel_fontsize = 14, yticklabel_fontsize = 14, annotation_fontsize = 14, bar_width = 0.2, fontsize_adjust=0, item_name=item_name, rank=True, rank_number=10, df_junior=df_junior, df_junior_faculty=df_junior_faculty, df_junior_school=df_junior_original, desired_order=desired_order)    
st.markdown("##")  ## 更大的间隔











  
