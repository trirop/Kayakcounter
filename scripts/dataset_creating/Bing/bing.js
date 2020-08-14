elements = document.querySelectorAll(’.mimg’)
var urls = [];
for (var i = 0; i < elements.length; i++) {
var url = elements[i].getAttribute(‘src’)
if (url&&url.includes(‘https’)) {
urls.push(url);
}
}
window.open(‘data:text/csv;charset=utf-8,’ + escape(urls.join(’\n’)));
