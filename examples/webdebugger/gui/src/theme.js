//Theme
import {grey100, grey500, grey600, grey700, grey800, lightBlack, orange600, darkBlack, white, cyan500} from 'material-ui/styles/colors';
import getMuiTheme from 'material-ui/styles/getMuiTheme';
import {fade} from 'material-ui/utils/colorManipulator';

const DavMuiTheme = getMuiTheme({
  palette: {
primary1Color: grey800,
primary2Color: grey700,
primary3Color: lightBlack,
accent1Color: orange600,
accent2Color: grey100,
accent3Color: grey500,
textColor: darkBlack,
alternateTextColor: white,
canvasColor: white,
borderColor: grey600,
disabledColor: fade(darkBlack, 0.3),
pickerHeaderColor: cyan500,
  }
});

export default DavMuiTheme;
