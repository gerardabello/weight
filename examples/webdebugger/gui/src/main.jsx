
'use strict';

//Temporal fix for 300ms delay
import injectTapEventPlugin from 'react-tap-event-plugin';
injectTapEventPlugin();

var React = require('react');
import { render } from 'react-dom'


import App from "./app.jsx";

import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';
import DavMuiTheme from "./theme.js";



render(
    <MuiThemeProvider muiTheme={DavMuiTheme}>
        <App/>
    </MuiThemeProvider>
,document.getElementById('react-root'));
