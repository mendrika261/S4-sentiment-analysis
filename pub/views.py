from django.http import JsonResponse
from django.shortcuts import render

from pub.models import Comment


def index(request):
    context = {}
    return render(request, 'index.html', context)


def sentiment_analysis(request):
    comment = Comment()
    comment.content = request.POST.get('comment')
    return JsonResponse({'positive': comment.is_positive()})


def about(request):
    comment = Comment()
    comment.content = request.POST.get('comment')
    return JsonResponse(comment.about())


def new_comment(request):
    Comment.objects.create(content=request.POST.get('comment'))
