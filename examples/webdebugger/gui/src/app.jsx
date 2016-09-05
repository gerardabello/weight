var React = require('react');


import Layers from "./components/layers.jsx";
import Train from "./components/train.jsx";

const styles = {

    main:{
        display: 'flex',
        alignItems: 'center',
        flexDirection: 'column',
    },
    container:{
        width: '100%',
    }
};

var App = React.createClass({

  getInitialState(){
        return({});
  },

	render: function() {
		return (
			<div>
        <Train/>
        <Layers/>
			</div>
		);
	}
});

export default App
