var React = require('react');


import Layer from './layer.jsx';

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

var Layers = React.createClass({

  getInitialState(){

        return({open: false});

  },

  componentDidMount(){

  if (this.state.ws) {
            return false;
        }
        let ws = new WebSocket("ws://localhost:8888/layers");

        ws.onopen = this.onOpen;
        ws.onclose = this.onClose;
        ws.onmessage = this.onMessage;

        ws.onerror = function(evt) {
            //print("ERROR: " + evt.data);
        }

        this.setState({ws: ws})

  },

  onMessage(evt){
    let data = JSON.parse(evt.data);
    this.setState({data: data});

  },

    onOpen(evt){
    this.setState({open: true});
  },

  onClose(evt){
    this.setState({open: false, ws: null});
  },

	render: function() {
    if (!this.state.open){
      return(<p>Socket closed</p>)
    }

		return (
			<div>
        {this.state.data.map(layer =>
          <Layer key={layer.ID} layer={layer}/>
        )}
			</div>
		);
	}
});

export default Layers
