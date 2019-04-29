from IPython.core.display import HTML

def format_notebook_wide():
    '''Changes format of Jupyter Notebook to a more readable format.
    Widens the display and changes the font.
    '''
    return HTML("""
                <style>
                div.cell {
                    margin-top:1em;
                    margin-bottom:1em;
                }
                div.text_cell_render h1 {
                    font-size: 1.6em;
                    line-height:1.2em;
                    text-align:center;
                }
                div.text_cell_render h2 {
                margin-bottom: -0.2em;
                }
                table tbody tr td:first-child,
                table tbody tr th:first-child,
                table thead tr th:first-child,
                table tbody tr td:nth-child(4),
                table thead tr th:nth-child(4) {
                    background-color: #edf4e8;
                }
                div.text_cell_render {
                    font-family: 'Garamond';
                    font-size:1.3em;
                    line-height:1.3em;
                    padding-left:3em;
                    padding-right:3em;
                }
                div#notebook-container    { width: 95%; }
                div#menubar-container     { width: 65%; }
                div#maintoolbar-container { width: 99%; }
                </style>
                """)

def hide_notebook_code_cells():
    '''Creates an HTML widget to hire all of the live code
    cells inside of a notebook.
    '''

    return HTML('''<script>
                code_show=true;
                function code_toggle() {
                 if (code_show){
                 $('div.input').hide();
                 } else {
                 $('div.input').show();
                 }
                 code_show = !code_show
                }
                $( document ).ready(code_toggle);
                </script>
                <form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
