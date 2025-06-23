document.addEventListener('DOMContentLoaded', function() {
    const currentPath = window.location.pathname;
    const links = {
        '/': 'home-link',
        '/analyze': 'analyze-link',
        '/faq': 'faq-link'
    };

    // Сначала удаляем класс active у всех ссылок
    Object.values(links).forEach(id => {
        const element = document.getElementById(id);
        if (element) element.classList.remove('active');
    });

    // Затем добавляем класс active только к текущей ссылке
    if (links.hasOwnProperty(currentPath)) {
        const activeElement = document.getElementById(links[currentPath]);
        if (activeElement) activeElement.classList.add('active');
    }
});