import os, sys
# os.chdir("F:/01Algorithms/webmap")
import http.server
import socketserver

# rootpath = os.path.abspath(os.getcwd())
rootpath = r"C:\gdrive\01Algorithms\Visualization\Digital-Earth\examples\webmap"
sys.path.append(rootpath + "/examples/webmap/")
datapath = os.path.join(rootpath, "data")
os.chdir(rootpath)

from functions import CreateMap

CreateMap(datapath)



class RequestsHandler(http.server.SimpleHTTPRequestHandler):
  """
  Handles http requests
  """
  def do_GET(self):
    if self.path == '/':
      self.path = 'src/RCS.html'
    return http.server.SimpleHTTPRequestHandler.do_GET(self)

handler_object = RequestsHandler

PORT = 80
my_server = socketserver.TCPServer(("", PORT), handler_object)
# Start the server
print("Server started at localhost:" + str(PORT))
my_server.serve_forever()



