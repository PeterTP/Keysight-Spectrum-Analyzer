# shop/urls.py

from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('display/', views.statistics_view, name='shop-statistics'),
    path('chart/sales/<str:year>/', views.compute_data, name='chart-sales'),
    path('chart/filter-options/', views.get_filter_options, name='chart-filter-options'),
]




