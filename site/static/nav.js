document.addEventListener('DOMContentLoaded', function() {
    const currentPath = window.location.pathname;
    
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.classList.remove('active');
    });
    
    if (currentPath === '/faq') {
        document.getElementById('faq-link').classList.add('active');
    } else {
        document.getElementById('home-link').classList.add('active');
    }
});