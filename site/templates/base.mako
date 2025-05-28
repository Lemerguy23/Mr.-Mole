<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${self.title()}</title>
    <link rel="stylesheet" href="/static/css_content/all_styles.css">
</head>
<body>
    <header class="header">
        <div class="logo">Mr.mole</div>
        <nav class="nav-links">
            <a href="/" id="home-link">Главная</a>
            <a href="/faq" id="faq-link">FAQ</a>
        </nav>
    </header>
    <main class="main-content">
        ${self.content()}
    </main>
    ${self.scripts()}
</body>
</html>

<%def name="title()">
Mr. Mole
</%def>

<%def name="scripts()">
    <script src="/static/scripts/nav.js"></script>
</%def>