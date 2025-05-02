<%inherit file="base.mako"/>

<%def name="title()">
Image Processor - FAQ
</%def>

<%def name="content()">
    <h1>Часто задаваемые вопросы</h1>
    
     <div class="faq-container">
        % for i, faq in enumerate(faq_items, 1):
        <div class="faq-item">
            <div class="faq-question">${i}. ${faq['question']}</div>
            <div class="faq-answer">
                <p>${faq['answer']}</p>
            </div>
        </div>
        % endfor
    </div>
</%def>

<%def name="scripts()">
    ${parent.scripts()}
    <script src="/static/scripts/faq.js"></script>
</%def>