from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('classifier/', include('classifier.urls')),
    path('admin/', admin.site.urls),
]
