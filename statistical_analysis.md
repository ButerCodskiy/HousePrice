# Профессиональный Эконометрический Анализ

Анализ проведен на логарифме цены `np.log1p(price)`. Использованы **робастные стандартные ошибки Уайта (HC3)** для корректировки гетероскедастичности. Категориальные признаки закодированы с удалением первой категории (`drop_first=True`) во избежание строгой мультиколлинеарности.

## 1. Топ-20 самых значимых предикторов
Отсортированы по t-статистике (надежности влияния на цену). Коэффициенты стандартизированных числовых признаков показывают изменение лог-цены при изменении признака на 1 стандартное отклонение.

|                                  |   Коэффициент |   Std Err (HC3) |   t-статистика |      p-value |
|:---------------------------------|--------------:|----------------:|---------------:|-------------:|
| room_type_Private room           |    -0.679478  |      0.00405397 |      -167.608  | 0            |
| room_type_Shared room            |    -1.04763   |      0.0163084  |       -64.2387 | 0            |
| availability_365                 |     0.0812964 |      0.00227439 |        35.7442 | 8.13538e-280 |
| center_distance                  |    -0.210242  |      0.0114834  |       -18.3083 | 7.10348e-75  |
| neighbourhood_Ridgewood          |    -0.536292  |      0.0309088  |       -17.3508 | 1.94613e-67  |
| neighbourhood_Breezy Point       |     1.33415   |      0.0781808  |        17.065  | 2.70499e-65  |
| year                             |    -0.0513135 |      0.00317131 |       -16.1805 | 6.92229e-59  |
| neighbourhood_Sunnyside          |    -0.483222  |      0.0317402  |       -15.2243 | 2.44096e-52  |
| neighbourhood_Midtown            |     0.20772   |      0.0152775  |        13.5965 | 4.20113e-42  |
| neighbourhood_Maspeth            |    -0.525444  |      0.0415342  |       -12.6509 | 1.10624e-36  |
| neighbourhood_East Elmhurst      |    -0.361909  |      0.0299171  |       -12.0971 | 1.09435e-33  |
| neighbourhood_Woodside           |    -0.422508  |      0.0358447  |       -11.7872 | 4.54525e-32  |
| neighbourhood_Astoria            |    -0.3087    |      0.0266507  |       -11.5832 | 5.01438e-31  |
| neighbourhood_Bedford-Stuyvesant |    -0.166466  |      0.0145458  |       -11.4443 | 2.51088e-30  |
| neighbourhood_Corona             |    -0.51342   |      0.0457243  |       -11.2286 | 2.95116e-29  |
| neighbourhood_Bushwick           |    -0.158473  |      0.0142382  |       -11.1301 | 8.95009e-29  |
| neighbourhood_Lower East Side    |    -0.195921  |      0.0179846  |       -10.8938 | 1.23324e-27  |
| neighbourhood_Borough Park       |    -0.295006  |      0.0276946  |       -10.6521 | 1.70445e-26  |
| neighbourhood_Hell's Kitchen     |     0.128551  |      0.0121222  |        10.6046 | 2.83598e-26  |
| neighbourhood_Chinatown          |    -0.237178  |      0.02262    |       -10.4853 | 1.00927e-25  |

## 2. Анализ Мультиколлинеарности (VIF)
Значение VIF > 10 указывает на сильную мультиколлинеарность. В нашей числовой выборке:

| Feature                        |     VIF |
|:-------------------------------|--------:|
| minimum_nights                 | 1.06043 |
| number_of_reviews              | 1.57184 |
| reviews_per_month              | 1.52023 |
| calculated_host_listings_count | 1.11266 |
| availability_365               | 1.15627 |
| center_distance                | 1.05559 |
| year                           | 2.71253 |
| month                          | 2.62083 |

## 3. Полный лог регрессии (с поправками HC3)
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.593
Model:                            OLS   Adj. R-squared:                  0.591
Method:                 Least Squares   F-statistic:                     333.9
Date:                Tue, 07 Jul 2026   Prob (F-statistic):               0.00
Time:                        09:04:12   Log-Likelihood:                -23693.
No. Observations:               47840   AIC:                         4.784e+04
Df Residuals:                   47611   BIC:                         4.985e+04
Df Model:                         228                                         
Covariance Type:                  HC3                                         
============================================================================================================
                                               coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------
const                                        5.3041      0.089     59.873      0.000       5.130       5.478
minimum_nights                              -0.0455      0.007     -6.633      0.000      -0.059      -0.032
number_of_reviews                           -0.0152      0.002     -7.787      0.000      -0.019      -0.011
reviews_per_month                            0.0091      0.002      3.987      0.000       0.005       0.014
calculated_host_listings_count              -0.0097      0.002     -4.647      0.000      -0.014      -0.006
availability_365                             0.0813      0.002     35.744      0.000       0.077       0.086
center_distance                             -0.2102      0.011    -18.308      0.000      -0.233      -0.188
year                                        -0.0513      0.003    -16.181      0.000      -0.058      -0.045
month                                        0.0234      0.003      7.906      0.000       0.018       0.029
neighbourhood_group_Brooklyn                -0.3653      0.087     -4.194      0.000      -0.536      -0.195
neighbourhood_group_Manhattan               -0.1537      0.088     -1.738      0.082      -0.327       0.020
neighbourhood_group_Queens                  -0.0419      0.082     -0.513      0.608      -0.202       0.118
neighbourhood_group_Staten Island           -0.2045      0.294     -0.696      0.486      -0.780       0.371
neighbourhood_Arden Heights                  0.0362      0.288      0.126      0.900      -0.529       0.601
neighbourhood_Arrochar                      -0.2113      0.303     -0.697      0.486      -0.805       0.383
neighbourhood_Arverne                        0.4319      0.062      6.976      0.000       0.311       0.553
neighbourhood_Astoria                       -0.3087      0.027    -11.583      0.000      -0.361      -0.256
neighbourhood_Bath Beach                    -0.0692      0.069     -0.998      0.318      -0.205       0.067
neighbourhood_Battery Park City             -0.0630      0.063     -1.006      0.315      -0.186       0.060
neighbourhood_Bay Ridge                     -0.0942      0.035     -2.723      0.006      -0.162      -0.026
neighbourhood_Bay Terrace                    0.4435      0.134      3.301      0.001       0.180       0.707
neighbourhood_Bay Terrace, Staten Island     0.1611      0.701      0.230      0.818      -1.213       1.535
neighbourhood_Baychester                     0.0646      0.121      0.533      0.594      -0.173       0.302
neighbourhood_Bayside                        0.0892      0.078      1.150      0.250      -0.063       0.241
neighbourhood_Bayswater                      0.2627      0.079      3.343      0.001       0.109       0.417
neighbourhood_Bedford-Stuyvesant            -0.1665      0.015    -11.444      0.000      -0.195      -0.138
neighbourhood_Belle Harbor                   0.5409      0.228      2.375      0.018       0.095       0.987
neighbourhood_Bellerose                      0.4289      0.130      3.311      0.001       0.175       0.683
neighbourhood_Belmont                       -0.1250      0.145     -0.863      0.388      -0.409       0.159
neighbourhood_Bensonhurst                   -0.1301      0.046     -2.852      0.004      -0.220      -0.041
neighbourhood_Bergen Beach                   0.0295      0.149      0.198      0.843      -0.263       0.322
neighbourhood_Boerum Hill                    0.0297      0.032      0.921      0.357      -0.034       0.093
neighbourhood_Borough Park                  -0.2950      0.028    -10.652      0.000      -0.349      -0.241
neighbourhood_Breezy Point                   1.3342      0.078     17.065      0.000       1.181       1.487
neighbourhood_Briarwood                     -0.0579      0.072     -0.799      0.424      -0.200       0.084
neighbourhood_Brighton Beach                 0.2139      0.046      4.626      0.000       0.123       0.305
neighbourhood_Bronxdale                     -0.3559      0.096     -3.718      0.000      -0.544      -0.168
neighbourhood_Brooklyn Heights              -0.0058      0.037     -0.159      0.874      -0.078       0.066
neighbourhood_Brownsville                   -0.1584      0.045     -3.486      0.000      -0.247      -0.069
neighbourhood_Bull's Head                   -0.1444      0.306     -0.472      0.637      -0.744       0.456
neighbourhood_Bushwick                      -0.1585      0.014    -11.130      0.000      -0.186      -0.131
neighbourhood_Cambria Heights                0.2305      0.092      2.504      0.012       0.050       0.411
neighbourhood_Canarsie                      -0.0100      0.037     -0.268      0.788      -0.083       0.063
neighbourhood_Carroll Gardens                0.0260      0.031      0.835      0.404      -0.035       0.087
neighbourhood_Castle Hill                   -0.4908      0.105     -4.668      0.000      -0.697      -0.285
neighbourhood_Castleton Corners              0.1832      0.515      0.356      0.722      -0.827       1.193
neighbourhood_Chelsea                        0.0653      0.015      4.314      0.000       0.036       0.095
neighbourhood_Chinatown                     -0.2372      0.023    -10.485      0.000      -0.282      -0.193
neighbourhood_City Island                    0.2280      0.117      1.949      0.051      -0.001       0.457
neighbourhood_Civic Center                  -0.2248      0.060     -3.718      0.000      -0.343      -0.106
neighbourhood_Claremont Village             -0.1656      0.124     -1.339      0.180      -0.408       0.077
neighbourhood_Clason Point                  -0.0404      0.144     -0.280      0.779      -0.323       0.242
neighbourhood_Clifton                       -0.1325      0.322     -0.411      0.681      -0.764       0.499
neighbourhood_Clinton Hill                  -0.0167      0.023     -0.728      0.467      -0.062       0.028
neighbourhood_Co-op City                     0.3109      0.083      3.769      0.000       0.149       0.473
neighbourhood_Cobble Hill                    0.0522      0.039      1.322      0.186      -0.025       0.130
neighbourhood_College Point                 -0.2707      0.098     -2.749      0.006      -0.464      -0.078
neighbourhood_Columbia St                   -0.1044      0.054     -1.939      0.052      -0.210       0.001
neighbourhood_Concord                       -0.3862      0.295     -1.307      0.191      -0.965       0.193
neighbourhood_Concourse                     -0.2812      0.100     -2.826      0.005      -0.476      -0.086
neighbourhood_Concourse Village             -0.2551      0.100     -2.542      0.011      -0.452      -0.058
neighbourhood_Coney Island                   0.3056      0.138      2.208      0.027       0.034       0.577
neighbourhood_Corona                        -0.5134      0.046    -11.229      0.000      -0.603      -0.424
neighbourhood_Crown Heights                 -0.1225      0.016     -7.509      0.000      -0.154      -0.091
neighbourhood_Cypress Hills                 -0.1246      0.039     -3.205      0.001      -0.201      -0.048
neighbourhood_DUMBO                          0.2431      0.068      3.581      0.000       0.110       0.376
neighbourhood_Ditmars Steinway              -0.3003      0.030     -9.965      0.000      -0.359      -0.241
neighbourhood_Dongan Hills                  -0.1246      0.315     -0.396      0.692      -0.741       0.492
neighbourhood_Douglaston                     0.1564      0.110      1.425      0.154      -0.059       0.371
neighbourhood_Downtown Brooklyn              0.0025      0.043      0.058      0.954      -0.081       0.086
neighbourhood_Dyker Heights                 -0.1334      0.133     -1.005      0.315      -0.394       0.127
neighbourhood_East Elmhurst                 -0.3619      0.030    -12.097      0.000      -0.421      -0.303
neighbourhood_East Flatbush                 -0.1478      0.023     -6.485      0.000      -0.192      -0.103
neighbourhood_East Harlem                    0.0505      0.019      2.723      0.006       0.014       0.087
neighbourhood_East Morrisania               -0.0109      0.175     -0.062      0.951      -0.354       0.332
neighbourhood_East New York                 -0.0929      0.033     -2.850      0.004      -0.157      -0.029
neighbourhood_East Village                  -0.1110      0.014     -7.699      0.000      -0.139      -0.083
neighbourhood_Eastchester                    0.3796      0.148      2.570      0.010       0.090       0.669
neighbourhood_Edenwald                       0.1174      0.149      0.786      0.432      -0.175       0.410
neighbourhood_Edgemere                       0.1867      0.159      1.174      0.240      -0.125       0.498
neighbourhood_Elmhurst                      -0.3720      0.036    -10.374      0.000      -0.442      -0.302
neighbourhood_Eltingville                    0.4544      0.611      0.743      0.457      -0.744       1.653
neighbourhood_Emerson Hill                  -0.1175      0.379     -0.310      0.757      -0.861       0.626
neighbourhood_Far Rockaway                   0.2839      0.112      2.542      0.011       0.065       0.503
neighbourhood_Fieldston                      0.1080      0.122      0.888      0.375      -0.130       0.346
neighbourhood_Financial District            -0.0889      0.021     -4.246      0.000      -0.130      -0.048
neighbourhood_Flatbush                      -0.1576      0.020     -7.803      0.000      -0.197      -0.118
neighbourhood_Flatiron District              0.1350      0.043      3.150      0.002       0.051       0.219
neighbourhood_Flatlands                      0.0159      0.053      0.299      0.765      -0.088       0.120
neighbourhood_Flushing                      -0.0656      0.024     -2.685      0.007      -0.114      -0.018
neighbourhood_Fordham                       -0.1076      0.095     -1.133      0.257      -0.294       0.079
neighbourhood_Forest Hills                  -0.1463      0.041     -3.556      0.000      -0.227      -0.066
neighbourhood_Fort Greene                   -0.0181      0.024     -0.756      0.450      -0.065       0.029
neighbourhood_Fort Hamilton                 -0.0257      0.051     -0.502      0.616      -0.126       0.075
neighbourhood_Fresh Meadows                  0.0187      0.087      0.214      0.831      -0.153       0.190
neighbourhood_Glendale                      -0.4848      0.064     -7.611      0.000      -0.610      -0.360
neighbourhood_Gowanus                        0.0626      0.030      2.062      0.039       0.003       0.122
neighbourhood_Gramercy                      -0.0521      0.022     -2.366      0.018      -0.095      -0.009
neighbourhood_Graniteville                  -0.4345      0.442     -0.983      0.326      -1.301       0.432
neighbourhood_Grant City                    -0.5183      0.293     -1.771      0.077      -1.092       0.055
neighbourhood_Gravesend                     -0.0192      0.051     -0.374      0.709      -0.120       0.081
neighbourhood_Great Kills                    0.3621      0.360      1.005      0.315      -0.344       1.069
neighbourhood_Greenpoint                     0.0134      0.019      0.703      0.482      -0.024       0.051
neighbourhood_Greenwich Village              0.0244      0.023      1.056      0.291      -0.021       0.070
neighbourhood_Grymes Hill                    0.2208      0.319      0.693      0.488      -0.404       0.845
neighbourhood_Harlem                         0.0582      0.020      2.899      0.004       0.019       0.098
neighbourhood_Hell's Kitchen                 0.1286      0.012     10.605      0.000       0.105       0.152
neighbourhood_Highbridge                    -0.3263      0.128     -2.558      0.011      -0.576      -0.076
neighbourhood_Hollis                         0.2093      0.109      1.929      0.054      -0.003       0.422
neighbourhood_Holliswood                     0.7494      0.284      2.635      0.008       0.192       1.307
neighbourhood_Howard Beach                  -0.1011      0.095     -1.060      0.289      -0.288       0.086
neighbourhood_Howland Hook                  -0.1201      0.307     -0.391      0.696      -0.723       0.482
neighbourhood_Huguenot                       0.3081      0.443      0.696      0.486      -0.559       1.175
neighbourhood_Hunts Point                   -0.5095      0.102     -5.018      0.000      -0.708      -0.310
neighbourhood_Inwood                         0.1308      0.040      3.237      0.001       0.052       0.210
neighbourhood_Jackson Heights               -0.3389      0.032    -10.463      0.000      -0.402      -0.275
neighbourhood_Jamaica                        0.0045      0.037      0.121      0.904      -0.069       0.078
neighbourhood_Jamaica Estates                0.0889      0.135      0.657      0.511      -0.177       0.354
neighbourhood_Jamaica Hills                  0.2471      0.145      1.704      0.088      -0.037       0.531
neighbourhood_Kensington                    -0.1398      0.035     -4.051      0.000      -0.207      -0.072
neighbourhood_Kew Gardens                   -0.0906      0.065     -1.384      0.166      -0.219       0.038
neighbourhood_Kew Gardens Hills             -0.1096      0.102     -1.075      0.282      -0.309       0.090
neighbourhood_Kingsbridge                    0.0099      0.092      0.108      0.914      -0.170       0.190
neighbourhood_Kips Bay                      -0.0859      0.021     -4.029      0.000      -0.128      -0.044
neighbourhood_Laurelton                      0.0857      0.126      0.678      0.498      -0.162       0.334
neighbourhood_Lighthouse Hill                0.3889      0.363      1.071      0.284      -0.323       1.101
neighbourhood_Little Italy                  -0.2112      0.038     -5.612      0.000      -0.285      -0.137
neighbourhood_Little Neck                    0.2735      0.155      1.760      0.078      -0.031       0.578
neighbourhood_Long Island City              -0.2571      0.032     -7.916      0.000      -0.321      -0.193
neighbourhood_Longwood                      -0.2236      0.099     -2.266      0.023      -0.417      -0.030
neighbourhood_Lower East Side               -0.1959      0.018    -10.894      0.000      -0.231      -0.161
neighbourhood_Manhattan Beach                0.1293      0.169      0.766      0.444      -0.202       0.460
neighbourhood_Marble Hill                    0.1346      0.179      0.754      0.451      -0.215       0.485
neighbourhood_Mariners Harbor                0.0435      0.328      0.133      0.894      -0.599       0.686
neighbourhood_Maspeth                       -0.5254      0.042    -12.651      0.000      -0.607      -0.444
neighbourhood_Melrose                       -0.3730      0.145     -2.575      0.010      -0.657      -0.089
neighbourhood_Middle Village                -0.3641      0.065     -5.580      0.000      -0.492      -0.236
neighbourhood_Midland Beach                 -0.1300      0.316     -0.411      0.681      -0.750       0.490
neighbourhood_Midtown                        0.2077      0.015     13.596      0.000       0.178       0.238
neighbourhood_Midwood                       -0.0877      0.039     -2.243      0.025      -0.164      -0.011
neighbourhood_Mill Basin                     0.2579      0.368      0.701      0.483      -0.463       0.979
neighbourhood_Morningside Heights            0.0336      0.025      1.328      0.184      -0.016       0.083
neighbourhood_Morris Heights                -0.2591      0.161     -1.611      0.107      -0.574       0.056
neighbourhood_Morris Park                   -0.0823      0.146     -0.563      0.573      -0.369       0.204
neighbourhood_Morrisania                    -0.1644      0.148     -1.112      0.266      -0.454       0.125
neighbourhood_Mott Haven                    -0.3201      0.096     -3.344      0.001      -0.508      -0.132
neighbourhood_Mount Eden                    -0.4423      0.191     -2.317      0.020      -0.816      -0.068
neighbourhood_Mount Hope                    -0.3132      0.119     -2.632      0.008      -0.547      -0.080
neighbourhood_Murray Hill                    0.0337      0.019      1.734      0.083      -0.004       0.072
neighbourhood_Navy Yard                      0.0480      0.111      0.431      0.667      -0.170       0.266
neighbourhood_Neponsit                       0.8386      0.125      6.718      0.000       0.594       1.083
neighbourhood_New Brighton                  -0.0210      0.368     -0.057      0.955      -0.742       0.700
neighbourhood_New Dorp                      -0.5477      0.944     -0.580      0.562      -2.398       1.303
neighbourhood_New Dorp Beach                -0.3078      0.303     -1.016      0.310      -0.902       0.286
neighbourhood_New Springville                0.0432      0.303      0.142      0.887      -0.551       0.637
neighbourhood_NoHo                           0.1163      0.051      2.281      0.023       0.016       0.216
neighbourhood_Nolita                        -0.0446      0.027     -1.629      0.103      -0.098       0.009
neighbourhood_North Riverdale                0.2120      0.154      1.378      0.168      -0.090       0.514
neighbourhood_Norwood                        0.0201      0.117      0.171      0.864      -0.209       0.250
neighbourhood_Oakwood                        0.1709      0.285      0.599      0.549      -0.388       0.730
neighbourhood_Olinville                      0.1752      0.353      0.496      0.620      -0.517       0.867
neighbourhood_Ozone Park                    -0.3240      0.042     -7.758      0.000      -0.406      -0.242
neighbourhood_Park Slope                     0.0778      0.023      3.343      0.001       0.032       0.123
neighbourhood_Parkchester                   -0.1674      0.113     -1.485      0.138      -0.388       0.054
neighbourhood_Pelham Bay                     0.0278      0.161      0.173      0.863      -0.287       0.343
neighbourhood_Pelham Gardens                -0.1004      0.133     -0.757      0.449      -0.360       0.159
neighbourhood_Port Morris                   -0.3539      0.103     -3.430      0.001      -0.556      -0.152
neighbourhood_Port Richmond                 -0.1498      0.340     -0.441      0.659      -0.815       0.516
neighbourhood_Prince's Bay                   0.7870      0.409      1.926      0.054      -0.014       1.588
neighbourhood_Prospect Heights               0.0245      0.026      0.951      0.341      -0.026       0.075
neighbourhood_Prospect-Lefferts Gardens     -0.1598      0.020     -7.813      0.000      -0.200      -0.120
neighbourhood_Queens Village                 0.0090      0.060      0.150      0.880      -0.109       0.127
neighbourhood_Randall Manor                 -0.3522      0.301     -1.171      0.242      -0.942       0.237
neighbourhood_Red Hook                      -0.1152      0.052     -2.227      0.026      -0.217      -0.014
neighbourhood_Rego Park                     -0.3425      0.045     -7.631      0.000      -0.430      -0.255
neighbourhood_Richmond Hill                 -0.1940      0.043     -4.511      0.000      -0.278      -0.110
neighbourhood_Richmondtown                  -0.3096      1.865     -0.166      0.868      -3.965       3.346
neighbourhood_Ridgewood                     -0.5363      0.031    -17.351      0.000      -0.597      -0.476
neighbourhood_Riverdale                      0.1644      0.160      1.025      0.305      -0.150       0.479
neighbourhood_Rockaway Beach                 0.2595      0.053      4.879      0.000       0.155       0.364
neighbourhood_Roosevelt Island              -0.1415      0.042     -3.344      0.001      -0.224      -0.059
neighbourhood_Rosebank                      -0.0829      0.302     -0.274      0.784      -0.675       0.509
neighbourhood_Rosedale                       0.1539      0.056      2.772      0.006       0.045       0.263
neighbourhood_Rossville                      0.0833     11.281      0.007      0.994     -22.026      22.193
neighbourhood_Schuylerville                 -0.0469      0.135     -0.348      0.728      -0.311       0.217
neighbourhood_Sea Gate                       0.4536      0.185      2.448      0.014       0.090       0.817
neighbourhood_Sheepshead Bay                 0.0832      0.040      2.106      0.035       0.006       0.161
neighbourhood_Shore Acres                    0.2187      0.421      0.520      0.603      -0.606       1.043
neighbourhood_Silver Lake                   -0.2844      0.477     -0.596      0.551      -1.219       0.650
neighbourhood_SoHo                          -0.0022      0.028     -0.078      0.937      -0.058       0.053
neighbourhood_Soundview                     -0.4800      0.108     -4.429      0.000      -0.692      -0.268
neighbourhood_South Beach                    0.0838      0.346      0.242      0.809      -0.595       0.762
neighbourhood_South Ozone Park              -0.1182      0.066     -1.783      0.075      -0.248       0.012
neighbourhood_South Slope                    0.0773      0.027      2.850      0.004       0.024       0.130
neighbourhood_Springfield Gardens            0.1799      0.048      3.727      0.000       0.085       0.275
neighbourhood_Spuyten Duyvil                 0.2562      0.381      0.673      0.501      -0.490       1.002
neighbourhood_St. Albans                     0.0487      0.058      0.836      0.403      -0.066       0.163
neighbourhood_St. George                    -0.2166      0.289     -0.750      0.453      -0.783       0.350
neighbourhood_Stapleton                     -0.1850      0.294     -0.630      0.529      -0.760       0.390
neighbourhood_Stuyvesant Town               -0.1830      0.054     -3.404      0.001      -0.288      -0.078
neighbourhood_Sunnyside                     -0.4832      0.032    -15.224      0.000      -0.545      -0.421
neighbourhood_Sunset Park                   -0.1830      0.024     -7.633      0.000      -0.230      -0.136
neighbourhood_Theater District               0.1829      0.026      7.113      0.000       0.133       0.233
neighbourhood_Throgs Neck                   -0.0135      0.113     -0.119      0.905      -0.235       0.208
neighbourhood_Todt Hill                      0.2009      0.535      0.376      0.707      -0.848       1.249
neighbourhood_Tompkinsville                 -0.3329      0.289     -1.153      0.249      -0.899       0.233
neighbourhood_Tottenville                    0.8462      0.371      2.280      0.023       0.119       1.574
neighbourhood_Tremont                       -0.4510      0.140     -3.228      0.001      -0.725      -0.177
neighbourhood_Tribeca                        0.1669      0.035      4.724      0.000       0.098       0.236
neighbourhood_Two Bridges                   -0.2543      0.045     -5.711      0.000      -0.342      -0.167
neighbourhood_Unionport                      0.0590      0.244      0.242      0.809      -0.418       0.536
neighbourhood_University Heights            -0.1570      0.105     -1.492      0.136      -0.363       0.049
neighbourhood_Upper East Side                0.0573      0.013      4.346      0.000       0.031       0.083
neighbourhood_Upper West Side                0.1424      0.014      9.881      0.000       0.114       0.171
neighbourhood_Van Nest                      -0.1656      0.213     -0.779      0.436      -0.582       0.251
neighbourhood_Vinegar Hill                   0.1559      0.079      1.978      0.048       0.001       0.310
neighbourhood_Wakefield                      0.0806      0.104      0.775      0.438      -0.123       0.284
neighbourhood_Washington Heights             0.0422      0.029      1.457      0.145      -0.015       0.099
neighbourhood_West Brighton                 -0.1746      0.299     -0.584      0.559      -0.760       0.411
neighbourhood_West Farms                     0.1822      0.082      2.210      0.027       0.021       0.344
neighbourhood_West Village                   0.0314      0.018      1.790      0.074      -0.003       0.066
neighbourhood_Westchester Square            -0.2361      0.141     -1.673      0.094      -0.513       0.041
neighbourhood_Westerleigh                   -0.2211      0.310     -0.713      0.476      -0.829       0.387
neighbourhood_Whitestone                    -0.0507      0.170     -0.298      0.766      -0.384       0.283
neighbourhood_Williamsbridge                -0.0161      0.110     -0.146      0.884      -0.232       0.200
neighbourhood_Williamsburg                   0.0493      0.016      2.997      0.003       0.017       0.081
neighbourhood_Willowbrook                    0.7083      0.830      0.853      0.394      -0.919       2.336
neighbourhood_Windsor Terrace                0.0198      0.036      0.544      0.586      -0.051       0.091
neighbourhood_Woodhaven                     -0.4577      0.044    -10.465      0.000      -0.543      -0.372
neighbourhood_Woodlawn                      -0.0267      0.182     -0.147      0.883      -0.383       0.329
neighbourhood_Woodside                      -0.4225      0.036    -11.787      0.000      -0.493      -0.352
room_type_Private room                      -0.6795      0.004   -167.608      0.000      -0.687      -0.672
room_type_Shared room                       -1.0476      0.016    -64.239      0.000      -1.080      -1.016
==============================================================================
Omnibus:                     3315.974   Durbin-Watson:                   1.927
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             7367.913
Skew:                           0.452   Prob(JB):                         0.00
Kurtosis:                       4.697   Cond. No.                     1.48e+16
==============================================================================

Notes:
[1] Standard Errors are heteroscedasticity robust (HC3)
[2] The smallest eigenvalue is 4.78e-28. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
```
