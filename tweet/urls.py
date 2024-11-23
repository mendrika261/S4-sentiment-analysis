"""
URL configuration for tweet project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from pub.views import index, sentiment_analysis, new_comment, about

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index),
    path('sentiment_analysis/', sentiment_analysis, name='sentiment_analysis'),
    path('new_comment/', new_comment, name='new_comment'),
    path('about/', about, name='about'),
]
