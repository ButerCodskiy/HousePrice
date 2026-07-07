# Профессиональный Эконометрический Анализ

Анализ проведен на логарифме цены `np.log1p(price)`. Использованы **робастные стандартные ошибки Уайта (HC3)** для корректировки гетероскедастичности. Категориальные признаки закодированы с удалением первой категории (`drop_first=True`) во избежание строгой мультиколлинеарности.

## 1. Топ-20 самых значимых предикторов
Отсортированы по t-статистике (надежности влияния на цену). Коэффициенты стандартизированных числовых признаков показывают изменение лог-цены при изменении признака на 1 стандартное отклонение.

|                                  |   Коэффициент |   Std Err (HC3) |   t-статистика |     p-value |
|:---------------------------------|--------------:|----------------:|---------------:|------------:|
| room_type_Private room           |    -0.698554  |      0.00432699 |      -161.441  | 0           |
| room_type_Shared room            |    -1.06729   |      0.0173628  |       -61.4698 | 0           |
| availability_365                 |     0.096133  |      0.00249711 |        38.4977 | 0           |
| neighbourhood_Breezy Point       |     1.33821   |      0.0740468  |        18.0725 | 5.25129e-73 |
| center_distance                  |    -0.21783   |      0.0121991  |       -17.8562 | 2.58486e-71 |
| year                             |    -0.0599464 |      0.00344885 |       -17.3816 | 1.13781e-67 |
| neighbourhood_Ridgewood          |    -0.553324  |      0.0319778  |       -17.3034 | 4.43566e-67 |
| neighbourhood_Midtown            |     0.273049  |      0.0170634  |        16.0021 | 1.23565e-57 |
| neighbourhood_Sunnyside          |    -0.496755  |      0.0331464  |       -14.9867 | 8.97251e-51 |
| neighbourhood_Maspeth            |    -0.550382  |      0.0422374  |       -13.0307 | 8.18733e-39 |
| neighbourhood_Woodside           |    -0.43444   |      0.0364427  |       -11.9212 | 9.18084e-33 |
| neighbourhood_Corona             |    -0.53185   |      0.0459466  |       -11.5754 | 5.49208e-31 |
| neighbourhood_Astoria            |    -0.321083  |      0.0277943  |       -11.5521 | 7.20236e-31 |
| neighbourhood_Lower East Side    |    -0.216772  |      0.019221   |       -11.2779 | 1.68738e-29 |
| neighbourhood_Bedford-Stuyvesant |    -0.167526  |      0.0149162  |       -11.2311 | 2.86775e-29 |
| neighbourhood_Bushwick           |    -0.161112  |      0.0144512  |       -11.1487 | 7.26672e-29 |
| neighbourhood_Chinatown          |    -0.265835  |      0.0241094  |       -11.0262 | 2.85773e-28 |
| neighbourhood_Borough Park       |    -0.307497  |      0.0278915  |       -11.0248 | 2.90316e-28 |
| neighbourhood_Woodhaven          |    -0.473057  |      0.0440652  |       -10.7354 | 6.94248e-27 |
| neighbourhood_Jackson Heights    |    -0.3517    |      0.0329264  |       -10.6814 | 1.24363e-26 |

## 2. Анализ Мультиколлинеарности (VIF)
Значение VIF > 10 указывает на сильную мультиколлинеарность. В нашей числовой выборке:

| Feature                        |     VIF |
|:-------------------------------|--------:|
| minimum_nights                 | 1.06009 |
| number_of_reviews              | 1.56862 |
| reviews_per_month              | 1.51724 |
| calculated_host_listings_count | 1.11159 |
| availability_365               | 1.15089 |
| center_distance                | 1.05302 |
| year                           | 2.7332  |
| month                          | 2.64081 |

## 3. Полный лог регрессии (с поправками HC3)
```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.577
Model:                            OLS   Adj. R-squared:                  0.575
Method:                 Least Squares   F-statistic:                     310.5
Date:                Tue, 07 Jul 2026   Prob (F-statistic):               0.00
Time:                        08:16:25   Log-Likelihood:                -28039.
No. Observations:               48645   AIC:                         5.654e+04
Df Residuals:                   48414   BIC:                         5.857e+04
Df Model:                         230                                         
Covariance Type:                  HC3                                         
============================================================================================================
                                               coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------------------------------------
const                                        5.3385      0.090     59.455      0.000       5.163       5.515
minimum_nights                              -0.0511      0.007     -6.849      0.000      -0.066      -0.036
number_of_reviews                           -0.0207      0.002    -10.162      0.000      -0.025      -0.017
reviews_per_month                            0.0080      0.002      3.343      0.001       0.003       0.013
calculated_host_listings_count              -0.0160      0.002     -6.943      0.000      -0.021      -0.011
availability_365                             0.0961      0.002     38.498      0.000       0.091       0.101
center_distance                             -0.2178      0.012    -17.856      0.000      -0.242      -0.194
year                                        -0.0599      0.003    -17.382      0.000      -0.067      -0.053
month                                        0.0254      0.003      7.915      0.000       0.019       0.032
neighbourhood_group_Brooklyn                -0.3755      0.088     -4.260      0.000      -0.548      -0.203
neighbourhood_group_Manhattan               -0.1423      0.090     -1.583      0.113      -0.318       0.034
neighbourhood_group_Queens                  -0.0441      0.082     -0.537      0.591      -0.205       0.117
neighbourhood_group_Staten Island           -0.1282      0.114     -1.127      0.260      -0.351       0.095
neighbourhood_Arden Heights                 -0.0355      0.104     -0.342      0.732      -0.239       0.168
neighbourhood_Arrochar                      -0.2320      0.154     -1.507      0.132      -0.534       0.070
neighbourhood_Arverne                        0.4465      0.064      6.970      0.000       0.321       0.572
neighbourhood_Astoria                       -0.3211      0.028    -11.552      0.000      -0.376      -0.267
neighbourhood_Bath Beach                    -0.0814      0.070     -1.164      0.244      -0.219       0.056
neighbourhood_Battery Park City             -0.0536      0.070     -0.768      0.442      -0.190       0.083
neighbourhood_Bay Ridge                     -0.0758      0.041     -1.867      0.062      -0.155       0.004
neighbourhood_Bay Terrace                    0.4347      0.133      3.257      0.001       0.173       0.696
neighbourhood_Bay Terrace, Staten Island     0.0909      0.641      0.142      0.887      -1.165       1.347
neighbourhood_Baychester                     0.0666      0.122      0.546      0.585      -0.172       0.306
neighbourhood_Bayside                        0.0865      0.077      1.126      0.260      -0.064       0.237
neighbourhood_Bayswater                      0.2589      0.079      3.295      0.001       0.105       0.413
neighbourhood_Bedford-Stuyvesant            -0.1675      0.015    -11.231      0.000      -0.197      -0.138
neighbourhood_Belle Harbor                   0.5252      0.227      2.310      0.021       0.080       0.971
neighbourhood_Bellerose                      0.4197      0.129      3.253      0.001       0.167       0.673
neighbourhood_Belmont                       -0.1313      0.145     -0.908      0.364      -0.415       0.152
neighbourhood_Bensonhurst                   -0.1372      0.046     -2.986      0.003      -0.227      -0.047
neighbourhood_Bergen Beach                   0.0169      0.150      0.113      0.910      -0.277       0.311
neighbourhood_Boerum Hill                    0.0422      0.036      1.178      0.239      -0.028       0.112
neighbourhood_Borough Park                  -0.3075      0.028    -11.025      0.000      -0.362      -0.253
neighbourhood_Breezy Point                   1.3382      0.074     18.072      0.000       1.193       1.483
neighbourhood_Briarwood                     -0.0333      0.079     -0.422      0.673      -0.188       0.122
neighbourhood_Brighton Beach                 0.2077      0.047      4.442      0.000       0.116       0.299
neighbourhood_Bronxdale                     -0.3605      0.096     -3.760      0.000      -0.548      -0.173
neighbourhood_Brooklyn Heights               0.0453      0.043      1.057      0.291      -0.039       0.129
neighbourhood_Brownsville                   -0.1630      0.046     -3.574      0.000      -0.252      -0.074
neighbourhood_Bull's Head                   -0.2254      0.138     -1.638      0.101      -0.495       0.044
neighbourhood_Bushwick                      -0.1611      0.014    -11.149      0.000      -0.189      -0.133
neighbourhood_Cambria Heights                0.2340      0.093      2.523      0.012       0.052       0.416
neighbourhood_Canarsie                      -0.0003      0.040     -0.009      0.993      -0.080       0.079
neighbourhood_Carroll Gardens                0.0327      0.033      0.986      0.324      -0.032       0.098
neighbourhood_Castle Hill                   -0.5120      0.104     -4.931      0.000      -0.716      -0.309
neighbourhood_Castleton Corners              0.0832      0.438      0.190      0.849      -0.775       0.942
neighbourhood_Chelsea                        0.0659      0.017      3.945      0.000       0.033       0.099
neighbourhood_Chinatown                     -0.2658      0.024    -11.026      0.000      -0.313      -0.219
neighbourhood_City Island                    0.4323      0.180      2.406      0.016       0.080       0.784
neighbourhood_Civic Center                  -0.2235      0.069     -3.258      0.001      -0.358      -0.089
neighbourhood_Claremont Village             -0.1794      0.123     -1.461      0.144      -0.420       0.061
neighbourhood_Clason Point                  -0.0596      0.144     -0.415      0.678      -0.341       0.222
neighbourhood_Clifton                       -0.2293      0.175     -1.307      0.191      -0.573       0.114
neighbourhood_Clinton Hill                3.119e-05      0.025      0.001      0.999      -0.049       0.049
neighbourhood_Co-op City                     0.2981      0.082      3.632      0.000       0.137       0.459
neighbourhood_Cobble Hill                    0.1090      0.050      2.169      0.030       0.011       0.207
neighbourhood_College Point                 -0.2811      0.099     -2.852      0.004      -0.474      -0.088
neighbourhood_Columbia St                   -0.1257      0.054     -2.330      0.020      -0.232      -0.020
neighbourhood_Concord                       -0.4811      0.118     -4.079      0.000      -0.712      -0.250
neighbourhood_Concourse                     -0.2954      0.100     -2.961      0.003      -0.491      -0.100
neighbourhood_Concourse Village             -0.2678      0.101     -2.653      0.008      -0.466      -0.070
neighbourhood_Coney Island                   0.2998      0.138      2.177      0.029       0.030       0.570
neighbourhood_Corona                        -0.5319      0.046    -11.575      0.000      -0.622      -0.442
neighbourhood_Crown Heights                 -0.1222      0.017     -7.274      0.000      -0.155      -0.089
neighbourhood_Cypress Hills                 -0.1050      0.043     -2.465      0.014      -0.188      -0.022
neighbourhood_DUMBO                          0.2186      0.069      3.180      0.001       0.084       0.353
neighbourhood_Ditmars Steinway              -0.3071      0.032     -9.717      0.000      -0.369      -0.245
neighbourhood_Dongan Hills                  -0.2123      0.164     -1.296      0.195      -0.533       0.109
neighbourhood_Douglaston                     0.1579      0.111      1.426      0.154      -0.059       0.375
neighbourhood_Downtown Brooklyn             -0.0021      0.045     -0.047      0.963      -0.090       0.086
neighbourhood_Dyker Heights                 -0.1447      0.133     -1.090      0.276      -0.405       0.115
neighbourhood_East Elmhurst                 -0.3528      0.033    -10.589      0.000      -0.418      -0.287
neighbourhood_East Flatbush                 -0.1533      0.023     -6.535      0.000      -0.199      -0.107
neighbourhood_East Harlem                    0.0310      0.019      1.591      0.112      -0.007       0.069
neighbourhood_East Morrisania               -0.0187      0.179     -0.105      0.916      -0.369       0.331
neighbourhood_East New York                 -0.1032      0.033     -3.146      0.002      -0.167      -0.039
neighbourhood_East Village                  -0.1333      0.015     -8.676      0.000      -0.163      -0.103
neighbourhood_Eastchester                    0.3592      0.146      2.463      0.014       0.073       0.645
neighbourhood_Edenwald                       0.1087      0.149      0.730      0.465      -0.183       0.401
neighbourhood_Edgemere                       0.1785      0.159      1.120      0.263      -0.134       0.491
neighbourhood_Elmhurst                      -0.3815      0.036    -10.538      0.000      -0.452      -0.311
neighbourhood_Eltingville                    0.3734      0.548      0.681      0.496      -0.700       1.447
neighbourhood_Emerson Hill                  -0.2036      0.267     -0.761      0.446      -0.728       0.321
neighbourhood_Far Rockaway                   0.4369      0.146      3.001      0.003       0.152       0.722
neighbourhood_Fieldston                      0.1098      0.124      0.884      0.377      -0.134       0.353
neighbourhood_Financial District            -0.0947      0.023     -4.136      0.000      -0.140      -0.050
neighbourhood_Flatbush                      -0.1537      0.021     -7.279      0.000      -0.195      -0.112
neighbourhood_Flatiron District              0.1802      0.052      3.481      0.000       0.079       0.282
neighbourhood_Flatlands                      0.0206      0.056      0.370      0.711      -0.089       0.130
neighbourhood_Flushing                      -0.0587      0.026     -2.274      0.023      -0.109      -0.008
neighbourhood_Fordham                       -0.1104      0.095     -1.161      0.246      -0.297       0.076
neighbourhood_Forest Hills                  -0.1304      0.045     -2.872      0.004      -0.219      -0.041
neighbourhood_Fort Greene                   -0.0196      0.025     -0.776      0.438      -0.069       0.030
neighbourhood_Fort Hamilton                 -0.0392      0.052     -0.759      0.448      -0.140       0.062
neighbourhood_Fort Wadsworth                 1.5098      0.442      3.418      0.001       0.644       2.376
neighbourhood_Fresh Meadows                  0.0181      0.087      0.208      0.835      -0.152       0.188
neighbourhood_Glendale                      -0.5039      0.064     -7.858      0.000      -0.630      -0.378
neighbourhood_Gowanus                        0.0554      0.031      1.768      0.077      -0.006       0.117
neighbourhood_Gramercy                      -0.0588      0.024     -2.410      0.016      -0.107      -0.011
neighbourhood_Graniteville                  -0.5219      0.341     -1.530      0.126      -1.190       0.146
neighbourhood_Grant City                    -0.6193      0.113     -5.486      0.000      -0.841      -0.398
neighbourhood_Gravesend                     -0.0247      0.052     -0.478      0.632      -0.126       0.076
neighbourhood_Great Kills                    0.2872      0.239      1.200      0.230      -0.182       0.756
neighbourhood_Greenpoint                     0.0080      0.020      0.407      0.684      -0.030       0.046
neighbourhood_Greenwich Village              0.0142      0.025      0.560      0.576      -0.036       0.064
neighbourhood_Grymes Hill                    0.1182      0.169      0.698      0.485      -0.214       0.450
neighbourhood_Harlem                         0.0500      0.021      2.337      0.019       0.008       0.092
neighbourhood_Hell's Kitchen                 0.1187      0.013      9.177      0.000       0.093       0.144
neighbourhood_Highbridge                    -0.3327      0.129     -2.586      0.010      -0.585      -0.081
neighbourhood_Hollis                         0.2068      0.108      1.917      0.055      -0.005       0.418
neighbourhood_Holliswood                     0.7439      0.289      2.577      0.010       0.178       1.310
neighbourhood_Howard Beach                  -0.1237      0.096     -1.294      0.196      -0.311       0.064
neighbourhood_Howland Hook                  -0.2246      0.164     -1.371      0.170      -0.546       0.096
neighbourhood_Huguenot                       0.2240      0.360      0.623      0.534      -0.481       0.929
neighbourhood_Hunts Point                   -0.5231      0.102     -5.146      0.000      -0.722      -0.324
neighbourhood_Inwood                         0.1171      0.042      2.793      0.005       0.035       0.199
neighbourhood_Jackson Heights               -0.3517      0.033    -10.681      0.000      -0.416      -0.287
neighbourhood_Jamaica                        0.0084      0.038      0.219      0.827      -0.066       0.083
neighbourhood_Jamaica Estates                0.2037      0.183      1.115      0.265      -0.154       0.562
neighbourhood_Jamaica Hills                  0.2420      0.142      1.709      0.087      -0.036       0.520
neighbourhood_Kensington                    -0.1389      0.035     -3.943      0.000      -0.208      -0.070
neighbourhood_Kew Gardens                   -0.1069      0.064     -1.667      0.095      -0.233       0.019
neighbourhood_Kew Gardens Hills             -0.1224      0.101     -1.214      0.225      -0.320       0.075
neighbourhood_Kingsbridge                    0.0075      0.092      0.081      0.935      -0.173       0.188
neighbourhood_Kips Bay                      -0.0868      0.024     -3.668      0.000      -0.133      -0.040
neighbourhood_Laurelton                      0.0846      0.127      0.666      0.506      -0.165       0.334
neighbourhood_Lighthouse Hill                0.2914      0.219      1.328      0.184      -0.139       0.721
neighbourhood_Little Italy                  -0.2044      0.043     -4.725      0.000      -0.289      -0.120
neighbourhood_Little Neck                    0.2817      0.151      1.867      0.062      -0.014       0.577
neighbourhood_Long Island City              -0.2736      0.034     -8.088      0.000      -0.340      -0.207
neighbourhood_Longwood                      -0.1966      0.107     -1.841      0.066      -0.406       0.013
neighbourhood_Lower East Side               -0.2168      0.019    -11.278      0.000      -0.254      -0.179
neighbourhood_Manhattan Beach                0.1197      0.170      0.704      0.481      -0.213       0.453
neighbourhood_Marble Hill                    0.1203      0.180      0.669      0.504      -0.232       0.473
neighbourhood_Mariners Harbor               -0.0475      0.179     -0.265      0.791      -0.398       0.303
neighbourhood_Maspeth                       -0.5504      0.042    -13.031      0.000      -0.633      -0.468
neighbourhood_Melrose                       -0.3891      0.147     -2.646      0.008      -0.677      -0.101
neighbourhood_Middle Village                -0.3904      0.065     -6.023      0.000      -0.517      -0.263
neighbourhood_Midland Beach                 -0.2389      0.166     -1.436      0.151      -0.565       0.087
neighbourhood_Midtown                        0.2730      0.017     16.002      0.000       0.240       0.306
neighbourhood_Midwood                       -0.0938      0.039     -2.391      0.017      -0.171      -0.017
neighbourhood_Mill Basin                     0.2278      0.363      0.627      0.531      -0.485       0.940
neighbourhood_Morningside Heights            0.0189      0.026      0.716      0.474      -0.033       0.071
neighbourhood_Morris Heights                -0.2726      0.160     -1.708      0.088      -0.585       0.040
neighbourhood_Morris Park                   -0.0805      0.147     -0.548      0.584      -0.368       0.207
neighbourhood_Morrisania                    -0.1821      0.148     -1.234      0.217      -0.471       0.107
neighbourhood_Mott Haven                    -0.3329      0.096     -3.458      0.001      -0.522      -0.144
neighbourhood_Mount Eden                    -0.4416      0.188     -2.350      0.019      -0.810      -0.073
neighbourhood_Mount Hope                    -0.3202      0.118     -2.705      0.007      -0.552      -0.088
neighbourhood_Murray Hill                    0.0304      0.022      1.386      0.166      -0.013       0.073
neighbourhood_Navy Yard                      0.0295      0.113      0.260      0.795      -0.193       0.252
neighbourhood_Neponsit                       0.8177      0.115      7.128      0.000       0.593       1.043
neighbourhood_New Brighton                  -0.1059      0.244     -0.433      0.665      -0.585       0.373
neighbourhood_New Dorp                      -0.6440      1.912     -0.337      0.736      -4.392       3.104
neighbourhood_New Dorp Beach                -0.4019      0.135     -2.971      0.003      -0.667      -0.137
neighbourhood_New Springville               -0.0315      0.134     -0.234      0.815      -0.295       0.232
neighbourhood_NoHo                           0.1120      0.054      2.075      0.038       0.006       0.218
neighbourhood_Nolita                        -0.0702      0.029     -2.441      0.015      -0.127      -0.014
neighbourhood_North Riverdale                0.2202      0.157      1.398      0.162      -0.088       0.529
neighbourhood_Norwood                        0.0188      0.117      0.160      0.873      -0.211       0.249
neighbourhood_Oakwood                        0.0856      0.088      0.973      0.330      -0.087       0.258
neighbourhood_Olinville                      0.1791      0.355      0.504      0.614      -0.517       0.875
neighbourhood_Ozone Park                    -0.3403      0.042     -8.118      0.000      -0.422      -0.258
neighbourhood_Park Slope                     0.1007      0.026      3.929      0.000       0.050       0.151
neighbourhood_Parkchester                   -0.1702      0.113     -1.503      0.133      -0.392       0.052
neighbourhood_Pelham Bay                     0.0136      0.161      0.084      0.933      -0.302       0.329
neighbourhood_Pelham Gardens                -0.1014      0.132     -0.766      0.444      -0.361       0.158
neighbourhood_Port Morris                   -0.3699      0.104     -3.568      0.000      -0.573      -0.167
neighbourhood_Port Richmond                 -0.2438      0.206     -1.183      0.237      -0.648       0.160
neighbourhood_Prince's Bay                   0.7086      0.301      2.357      0.018       0.119       1.298
neighbourhood_Prospect Heights               0.0351      0.028      1.268      0.205      -0.019       0.089
neighbourhood_Prospect-Lefferts Gardens     -0.1508      0.022     -6.888      0.000      -0.194      -0.108
neighbourhood_Queens Village                 0.0090      0.061      0.148      0.883      -0.110       0.128
neighbourhood_Randall Manor                 -0.4575      0.129     -3.554      0.000      -0.710      -0.205
neighbourhood_Red Hook                      -0.1154      0.055     -2.102      0.036      -0.223      -0.008
neighbourhood_Rego Park                     -0.3538      0.045     -7.888      0.000      -0.442      -0.266
neighbourhood_Richmond Hill                 -0.2053      0.043     -4.747      0.000      -0.290      -0.121
neighbourhood_Richmondtown                  -0.4099      0.564     -0.727      0.467      -1.515       0.696
neighbourhood_Ridgewood                     -0.5533      0.032    -17.303      0.000      -0.616      -0.491
neighbourhood_Riverdale                      0.6380      0.367      1.737      0.082      -0.082       1.358
neighbourhood_Rockaway Beach                 0.2762      0.058      4.763      0.000       0.163       0.390
neighbourhood_Roosevelt Island              -0.1672      0.042     -3.939      0.000      -0.250      -0.084
neighbourhood_Rosebank                      -0.1803      0.133     -1.356      0.175      -0.441       0.080
neighbourhood_Rosedale                       0.1599      0.056      2.864      0.004       0.050       0.269
neighbourhood_Rossville                      0.0125      0.906      0.014      0.989      -1.763       1.788
neighbourhood_Schuylerville                 -0.0419      0.134     -0.313      0.754      -0.304       0.221
neighbourhood_Sea Gate                       0.4531      0.191      2.367      0.018       0.078       0.828
neighbourhood_Sheepshead Bay                 0.0935      0.043      2.197      0.028       0.010       0.177
neighbourhood_Shore Acres                    0.1185      0.324      0.365      0.715      -0.517       0.754
neighbourhood_Silver Lake                   -0.3645      0.427     -0.854      0.393      -1.201       0.472
neighbourhood_SoHo                           0.0512      0.032      1.620      0.105      -0.011       0.113
neighbourhood_Soundview                     -0.4847      0.109     -4.432      0.000      -0.699      -0.270
neighbourhood_South Beach                   -0.0015      0.215     -0.007      0.994      -0.424       0.421
neighbourhood_South Ozone Park              -0.1249      0.067     -1.867      0.062      -0.256       0.006
neighbourhood_South Slope                    0.0636      0.027      2.332      0.020       0.010       0.117
neighbourhood_Springfield Gardens            0.1838      0.049      3.773      0.000       0.088       0.279
neighbourhood_Spuyten Duyvil                 0.2467      0.382      0.646      0.518      -0.502       0.995
neighbourhood_St. Albans                     0.0645      0.061      1.062      0.288      -0.055       0.184
neighbourhood_St. George                    -0.2798      0.106     -2.634      0.008      -0.488      -0.072
neighbourhood_Stapleton                     -0.2850      0.112     -2.544      0.011      -0.505      -0.065
neighbourhood_Stuyvesant Town               -0.1674      0.074     -2.267      0.023      -0.312      -0.023
neighbourhood_Sunnyside                     -0.4968      0.033    -14.987      0.000      -0.562      -0.432
neighbourhood_Sunset Park                   -0.1872      0.025     -7.611      0.000      -0.235      -0.139
neighbourhood_Theater District               0.2105      0.030      6.969      0.000       0.151       0.270
neighbourhood_Throgs Neck                   -0.0195      0.114     -0.171      0.864      -0.242       0.203
neighbourhood_Todt Hill                      0.1123      0.463      0.242      0.809      -0.796       1.021
neighbourhood_Tompkinsville                 -0.4281      0.100     -4.287      0.000      -0.624      -0.232
neighbourhood_Tottenville                    0.7612      0.253      3.012      0.003       0.266       1.256
neighbourhood_Tremont                       -0.4533      0.140     -3.234      0.001      -0.728      -0.179
neighbourhood_Tribeca                        0.2372      0.041      5.751      0.000       0.156       0.318
neighbourhood_Two Bridges                   -0.2932      0.045     -6.491      0.000      -0.382      -0.205
neighbourhood_Unionport                      0.0358      0.241      0.149      0.882      -0.437       0.509
neighbourhood_University Heights            -0.1586      0.107     -1.484      0.138      -0.368       0.051
neighbourhood_Upper East Side                0.0439      0.014      3.085      0.002       0.016       0.072
neighbourhood_Upper West Side                0.1449      0.016      9.280      0.000       0.114       0.175
neighbourhood_Van Nest                      -0.1729      0.207     -0.836      0.403      -0.578       0.233
neighbourhood_Vinegar Hill                   0.1637      0.082      1.999      0.046       0.003       0.324
neighbourhood_Wakefield                      0.0791      0.104      0.761      0.446      -0.125       0.283
neighbourhood_Washington Heights             0.0334      0.031      1.085      0.278      -0.027       0.094
neighbourhood_West Brighton                 -0.2751      0.125     -2.197      0.028      -0.521      -0.030
neighbourhood_West Farms                     0.1609      0.083      1.938      0.053      -0.002       0.324
neighbourhood_West Village                   0.0407      0.020      2.020      0.043       0.001       0.080
neighbourhood_Westchester Square            -0.0422      0.248     -0.171      0.865      -0.528       0.443
neighbourhood_Westerleigh                   -0.3081      0.130     -2.373      0.018      -0.563      -0.054
neighbourhood_Whitestone                    -0.0586      0.167     -0.351      0.725      -0.386       0.268
neighbourhood_Williamsbridge                -0.0238      0.110     -0.217      0.829      -0.240       0.192
neighbourhood_Williamsburg                   0.0491      0.017      2.894      0.004       0.016       0.082
neighbourhood_Willowbrook                    0.5897      0.905      0.651      0.515      -1.185       2.364
neighbourhood_Windsor Terrace                0.0058      0.036      0.160      0.873      -0.066       0.077
neighbourhood_Woodhaven                     -0.4731      0.044    -10.735      0.000      -0.559      -0.387
neighbourhood_Woodlawn                      -0.0128      0.183     -0.070      0.944      -0.372       0.347
neighbourhood_Woodrow                        2.1935      1.956      1.122      0.262      -1.640       6.027
neighbourhood_Woodside                      -0.4344      0.036    -11.921      0.000      -0.506      -0.363
room_type_Private room                      -0.6986      0.004   -161.441      0.000      -0.707      -0.690
room_type_Shared room                       -1.0673      0.017    -61.470      0.000      -1.101      -1.033
==============================================================================
Omnibus:                     6852.783   Durbin-Watson:                   1.905
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            18930.395
Skew:                           0.771   Prob(JB):                         0.00
Kurtosis:                       5.638   Cond. No.                     3.46e+16
==============================================================================

Notes:
[1] Standard Errors are heteroscedasticity robust (HC3)
[2] The smallest eigenvalue is 8.9e-29. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
```
