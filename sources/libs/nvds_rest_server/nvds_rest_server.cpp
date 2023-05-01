/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nvds_rest_server.h"

#include <unistd.h>

#include <iostream>
#include <memory>

#include "CivetServer.h"
#include "nvds_parse.h"

#define UNKNOWN_STRING "unknown"
#define EMPTY_STRING ""

enum NvDsErrorCode {
    NoError = 0,
    CameraUnauthorizedError = 0x1F, // HTTP error code : 403
    ClientUnauthorizedError,        // HTTP error code : 401
    InvalidParameterError,          // HTTP error code : 400
    CameraNotFoundError,            // HTTP error code : 404
    MethodNotAllowedError,          // HTTP error code : 405
    DeviceRequestTimeoutError,      // HTTP error code : 408
    CommunicationError,             // HTTP error code : 500
    DSInternalError,                // HTTP error code : 500
    DSNotSupportedError,            // HTTP error code : 501
    DSInsufficientStorage,          // HTTP error code : 507
};

/* ---------------------------------------------------------------------------
**  http callback
** -------------------------------------------------------------------------*/
class NvDsRestServer : public CivetServer {
public:
    typedef std::function<NvDsErrorCode(const Json::Value &,
                                        const Json::Value &,
                                        Json::Value &,
                                        struct mg_connection *conn)>
        httpFunction;

    NvDsRestServer(const std::vector<std::string> &options);

    void addRequestHandler(std::map<std::string, httpFunction> &func);
};

std::pair<int, std::string> translateNvDsErrorCodeToCameraHttpErrorCode(NvDsErrorCode code);

NvDsErrorCode translateCameraHttpErrorCodeToNvDsErrorCode(int code);

bool iequals(const std::string &a, const std::string &b);

int log_message(const struct mg_connection *conn, const char *message);

const struct CivetCallbacks *getCivetCallbacks();

static bool log_api_info(const std::string &api_name);

NvDsErrorCode VersionInfo(Json::Value &response, struct mg_connection *conn);

NvDsErrorCode handleUpdateROI(const Json::Value &req_info,
                              const Json::Value &in,
                              Json::Value &response,
                              struct mg_connection *conn,
                              std::function<void(NvDsRoiInfo *roi_ctx, void *ctx)> roi_cb);

NvDsErrorCode handleDecDropFrameInterval(
    const Json::Value &req_info,
    const Json::Value &in,
    Json::Value &response,
    struct mg_connection *conn,
    std::function<void(NvDsDecInfo *dec_ctx, void *ctx)> dec_cb);

NvDsErrorCode handleAddStream(const Json::Value &req_info,
                              const Json::Value &in,
                              Json::Value &response,
                              struct mg_connection *conn,
                              std::function<void(NvDsStreamInfo *stream_ctx, void *ctx)> stream_cb);

NvDsErrorCode handleRemoveStream(
    const Json::Value &req_info,
    const Json::Value &in,
    Json::Value &response,
    struct mg_connection *conn,
    std::function<void(NvDsStreamInfo *stream_ctx, void *ctx)> stream_cb);

NvDsErrorCode handleInferInterval(
    const Json::Value &req_info,
    const Json::Value &in,
    Json::Value &response,
    struct mg_connection *conn,
    std::function<void(NvDsInferInfo *infer_ctx, void *ctx)> infer_cb);

std::pair<int, std::string> translateNvDsErrorCodeToCameraHttpErrorCode(NvDsErrorCode code)
{
    switch ((int)code) {
    case NoError:
        return std::make_pair(200, "OK");
    case CameraUnauthorizedError:
        return std::make_pair(403, "Forbidden");
    case ClientUnauthorizedError:
        return std::make_pair(401, "Unauthorized");
    case InvalidParameterError:
        return std::make_pair(400, "Bad Request");
    case CameraNotFoundError:
        return std::make_pair(404, "Not Found");
    case MethodNotAllowedError:
        return std::make_pair(405, "Method Not Allowed");
    case DeviceRequestTimeoutError:
        return std::make_pair(408, "Request Timout");
    case DSNotSupportedError:
        return std::make_pair(501, "Not Implemented");
    case DSInsufficientStorage:
        return std::make_pair(507, "Insufficient Storage");
    case CommunicationError:
    default:
        return std::make_pair(501, "Internal Server Error");
    }
}

NvDsErrorCode translateCameraHttpErrorCodeToNvDsErrorCode(int code)
{
    switch (code) {
    case 200:
        return NvDsErrorCode::NoError;
    case 400:
        return NvDsErrorCode::CameraUnauthorizedError;
    case 401:
        return NvDsErrorCode::CameraUnauthorizedError;
    case 404:
        return NvDsErrorCode::CameraNotFoundError;
    case 405:
        return NvDsErrorCode::MethodNotAllowedError;
    case 408:
        return NvDsErrorCode::DeviceRequestTimeoutError;
    case 501:
        return NvDsErrorCode::DSNotSupportedError;
    case 507:
        return NvDsErrorCode::DSInsufficientStorage;
    default:
        return NvDsErrorCode::DSInternalError;
    }
}

bool iequals(const std::string &a, const std::string &b)
{
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                      [](char str1, char str2) { return tolower(str1) == tolower(str2); });
}

int log_message(const struct mg_connection *conn, const char *message)
{
    fprintf(stderr, "%s\n", message);
    // LOG(verbose) << "HTTP SERVER: " << message << endl;
    return 0;
}

static struct CivetCallbacks _callbacks;
const struct CivetCallbacks *getCivetCallbacks()
{
    // memset(&_callbacks, 0, sizeof(_callbacks));
    _callbacks.log_message = &log_message;
    return &_callbacks;
}

static bool log_api_info(const std::string &api_name)
{
    if ((api_name == "/api/stream/status") || (api_name == "/api/stream/stats")) {
        return false;
    }
    return true;
}

/* ---------------------------------------------------------------------------
**  Civet HTTP callback
** -------------------------------------------------------------------------*/
class RequestHandler : public CivetHandler {
public:
    RequestHandler(std::string uri, NvDsRestServer::httpFunction &func) : m_uri(uri), m_func(func)
    {
    }

    bool handle(CivetServer *server, struct mg_connection *conn)
    {
        bool ret = false;
        Json::Value response;
        Json::Value req;
        NvDsErrorCode result;
        const struct mg_request_info *req_info = mg_get_request_info(conn);
        if (req_info == NULL) {
            std::cout << "req_info is NULL " << std::endl;
            return ret;
        }

        if (log_api_info(req_info->request_uri)) {
            std::cout << "uri:" << req_info->request_uri << std::endl;
            std::cout << "method:" << req_info->request_method << std::endl;
        }

        if (m_uri.back() != '*') {
            if (m_uri != req_info->request_uri) {
                std::cout << "Wrong API uri:" << req_info->request_uri
                          << " Please use correct uri: " << m_uri << std::endl;
                return ret;
            }
        }
        // read input
        Json::Value in;
        result = this->getInputMessage(req_info, conn, in);
        if (result == NvDsErrorCode::NoError) {
            req["url"] = req_info->request_uri;
            req["method"] = req_info->request_method;
            req["query"] = req_info->query_string != NULL ? req_info->query_string : "";
            req["remote_addr"] = req_info->remote_addr;
            req["remote_user"] = req_info->remote_user != NULL ? req_info->remote_user : "";
            // invoke API implementation
            result = m_func(req, in, response, conn);
        } else {
            response = in;
        }
        return httpResponseHandler(result, response, conn);
    }

    bool httpResponseHandler(NvDsErrorCode &result,
                             Json::Value &response,
                             struct mg_connection *conn)
    {
        if (result == NvDsErrorCode::NoError) {
            mg_printf(conn, "HTTP/1.1 200 OK\r\n");
        } else {
            std::pair<int, std::string> http_err_code =
                translateNvDsErrorCodeToCameraHttpErrorCode(result);
            std::string response = std::string("HTTP/1.1 ") + std::to_string(http_err_code.first) +
                                   " " + http_err_code.second;
            mg_printf(conn, "%s\r\n", response.c_str());
        }
        mg_printf(conn, "Access-Control-Allow-Origin: *\r\n");
        std::string content_type;
        if (response.isObject()) {
            content_type = response.get("content_type", "").asString();
        }
        std::string answer;
        if (content_type.empty() == false) {
            answer = response.get("data", "").asString();
            mg_printf(conn, "Content-Type: image/jpeg\r\n");
        } else {
            std::string ans(Json::writeString(m_writerBuilder, response));
            answer = ans;
            mg_printf(conn, "Content-Type: text/plain\r\n");
        }
        mg_printf(conn, "Content-Length: %zd\r\n", answer.size());
        mg_printf(conn, "Connection: close\r\n");
        mg_printf(conn, "\r\n");
        mg_write(conn, answer.c_str(), answer.size());
        return true;
    }

    bool handleGet(CivetServer *server, struct mg_connection *conn) { return handle(server, conn); }
    bool handlePost(CivetServer *server, struct mg_connection *conn)
    {
        return handle(server, conn);
    }
    bool handlePut(CivetServer *server, struct mg_connection *conn) { return handle(server, conn); }
    bool handleDelete(CivetServer *server, struct mg_connection *conn)
    {
        return handle(server, conn);
    }

private:
    std::string m_uri;
    NvDsRestServer::httpFunction m_func;
    Json::StreamWriterBuilder m_writerBuilder;
    Json::CharReaderBuilder m_readerBuilder;

    NvDsErrorCode getInputMessage(const struct mg_request_info *req_info,
                                  struct mg_connection *conn,
                                  Json::Value &out)
    {
        // Return if content length is zero otherwise procede to check content type
        if (req_info == NULL || conn == NULL) {
            out = Json::nullValue;
            std::string error_message = "Request Information is null";
            std::cout << error_message << std::endl;

            return NvDsErrorCode::InvalidParameterError;
        }
        long long tlen = req_info->content_length;
        if (tlen > 0) {
            std::string body;
            unsigned long long rlen;
            long long nlen = 0;
            char buf[1024];
            while (nlen < tlen) {
                rlen = tlen - nlen;
                if (rlen > sizeof(buf)) {
                    rlen = sizeof(buf);
                }
                rlen = mg_read(conn, buf, (size_t)rlen);
                if (rlen <= 0) {
                    break;
                }
                try {
                    body.append(buf, rlen);
                } catch (const std::exception &e) {
                    std::cout << "Exception while fetching content data" << std::endl;
                    break;
                }
                nlen += rlen;
            }
            // parse in
            std::unique_ptr<Json::CharReader> reader(m_readerBuilder.newCharReader());
            std::string errors;
            if (!reader->parse(body.c_str(), body.c_str() + body.size(), &out, &errors)) {
                out = Json::nullValue;
                std::string error_message = std::string("Received unknown message:") + body +
                                            std::string(" errors:") + errors;
                std::cout << error_message << std::endl;
                return NvDsErrorCode::InvalidParameterError;
            }
        }
        return NvDsErrorCode::NoError;
    }
};

/* ---------------------------------------------------------------------------
**  Constructor
** -------------------------------------------------------------------------*/
NvDsRestServer::NvDsRestServer(const std::vector<std::string> &options)
    : CivetServer(options, getCivetCallbacks())
{
    std::cout << "Civetweb version: v" << mg_version() << std::endl;
}

NvDsErrorCode VersionInfo(Json::Value &response, struct mg_connection *conn)
{
    NvDsErrorCode ret = NvDsErrorCode::NoError;
    response["version"] = "DeepStream-SDK 6.2";
    return ret;
}

void __attribute__((constructor)) nvds_rest_server_init(void);
void __attribute__((destructor)) nvds_rest_server_deinit(void);

void __attribute__((constructor)) nvds_rest_server_init(void)
{
    mg_init_library(0);
}
void __attribute__((destructor)) nvds_rest_server_deinit(void)
{
    mg_exit_library();
}
void nvds_rest_server_stop(NvDsRestServer *handler)
{
    std::cout << "Stopping the server..!! \n";

    if (handler) {
        delete handler;
    }
}

NvDsErrorCode handleInferInterval(const Json::Value &req_info,
                                  const Json::Value &in,
                                  Json::Value &response,
                                  struct mg_connection *conn,
                                  std::function<void(NvDsInferInfo *infer_ctx, void *ctx)> infer_cb)
{
    NvDsErrorCode ret = NvDsErrorCode::NoError;
    const std::string request_api = req_info.get("url", EMPTY_STRING).asString();
    const std::string request_method = req_info.get("method", UNKNOWN_STRING).asString();
    const std::string query_string = req_info.get("query", EMPTY_STRING).asString();

    if (request_api.empty() || request_method == UNKNOWN_STRING) {
        std::cout << "Malformed HTTP request" << std::endl;
        return NvDsErrorCode::InvalidParameterError;
    }

    if (iequals(request_method, "get")) {
    }

    if (iequals(request_method, "post")) {
        NvDsInferInfo infer_info;
        NvDsResponseInfo res_info;

        void *custom_ctx;

        if (nvds_rest_infer_parse(in, &infer_info) && (infer_cb)) {
            infer_cb(&infer_info, &custom_ctx);
            if (infer_info.status == INFER_INTERVAL_UPDATE_SUCCESS)
                res_info.status = "INFER_INTERVAL_UPDATE_SUCCESS";
            else
                res_info.status = "INFER_INTERVAL_UPDATE_FAIL";
        } else {
            res_info.status = "INFER_INTERVAL_UPDATE_FAIL";
        }

        res_info.reason = "NA";

        response["status"] = res_info.status;
        response["reason"] = res_info.reason;
    }

    return ret;
}

NvDsErrorCode handleDecDropFrameInterval(
    const Json::Value &req_info,
    const Json::Value &in,
    Json::Value &response,
    struct mg_connection *conn,
    std::function<void(NvDsDecInfo *dec_ctx, void *ctx)> dec_cb)
{
    NvDsErrorCode ret = NvDsErrorCode::NoError;
    const std::string request_api = req_info.get("url", EMPTY_STRING).asString();
    const std::string request_method = req_info.get("method", UNKNOWN_STRING).asString();
    const std::string query_string = req_info.get("query", EMPTY_STRING).asString();

    if (request_api.empty() || request_method == UNKNOWN_STRING) {
        std::cout << "Malformed HTTP request" << std::endl;
        return NvDsErrorCode::InvalidParameterError;
    }

    if (iequals(request_method, "get")) {
    }

    if (iequals(request_method, "post")) {
        NvDsDecInfo dec_info;
        NvDsResponseInfo res_info;

        void *custom_ctx;

        if (nvds_rest_dec_parse(in, &dec_info) && (dec_cb)) {
            dec_cb(&dec_info, &custom_ctx);
            if (dec_info.status == DROP_FRAME_INTERVAL_UPDATE_SUCCESS)
                res_info.status = "DROP_FRAME_INTERVAL_UPDATE_SUCCESS";
            else
                res_info.status = "DROP_FRAME_INTERVAL_UPDATE_FAIL";
        } else {
            res_info.status = "DROP_FRAME_INTERVAL_UPDATE_FAIL";
        }

        res_info.reason = "NA";

        response["status"] = res_info.status;
        response["reason"] = res_info.reason;
    }

    return ret;
}

NvDsErrorCode handleUpdateROI(const Json::Value &req_info,
                              const Json::Value &in,
                              Json::Value &response,
                              struct mg_connection *conn,
                              std::function<void(NvDsRoiInfo *roi_ctx, void *ctx)> roi_cb)
{
    NvDsErrorCode ret = NvDsErrorCode::NoError;
    const std::string request_api = req_info.get("url", EMPTY_STRING).asString();
    const std::string request_method = req_info.get("method", UNKNOWN_STRING).asString();
    const std::string query_string = req_info.get("query", EMPTY_STRING).asString();

    if (request_api.empty() || request_method == UNKNOWN_STRING) {
        std::cout << "Malformed HTTP request" << std::endl;
        return NvDsErrorCode::InvalidParameterError;
    }

    if (iequals(request_method, "get")) {
    }

    if (iequals(request_method, "post")) {
        NvDsRoiInfo roi_info;
        NvDsResponseInfo res_info;

        void *custom_ctx;

        if (nvds_rest_roi_parse(in, &roi_info) && (roi_cb)) {
            roi_cb(&roi_info, &custom_ctx);
            if (roi_info.status == ROI_UPDATE_SUCCESS)
                res_info.status = "ROI_UPDATE_SUCCESS";
            else
                res_info.status = "ROI_UPDATE_FAIL";
        } else {
            res_info.status = "ROI_UPDATE_FAIL";
        }

        res_info.reason = "NA";

        response["status"] = res_info.status;
        response["reason"] = res_info.reason;
    }
    return ret;
}

NvDsErrorCode handleAddStream(const Json::Value &req_info,
                              const Json::Value &in,
                              Json::Value &response,
                              struct mg_connection *conn,
                              std::function<void(NvDsStreamInfo *stream_ctx, void *ctx)> stream_cb)
{
    NvDsErrorCode ret = NvDsErrorCode::NoError;
    const std::string request_api = req_info.get("url", EMPTY_STRING).asString();
    const std::string request_method = req_info.get("method", UNKNOWN_STRING).asString();
    const std::string query_string = req_info.get("query", EMPTY_STRING).asString();

    if (request_api.empty() || request_method == UNKNOWN_STRING) {
        std::cout << "Malformed HTTP request" << std::endl;
        return NvDsErrorCode::InvalidParameterError;
    }

    if (iequals(request_method, "get")) {
    }

    if (iequals(request_method, "post")) {
        NvDsStreamInfo stream_info;
        NvDsResponseInfo res_info;

        void *custom_ctx;

        if (nvds_rest_stream_parse(in, &stream_info) && (stream_cb)) {
            stream_cb(&stream_info, &custom_ctx);
            if (stream_info.status == STREAM_ADD_SUCCESS)
                res_info.status = "STREAM_ADD_SUCCESS";
            else
                res_info.status = "STREAM_ADD_FAIL";
        } else {
            res_info.status = "STREAM_ADD_FAIL";
        }
        res_info.reason = "NA";

        response["status"] = res_info.status;
        response["reason"] = res_info.reason;
    }
    return ret;
}

NvDsErrorCode handleRemoveStream(
    const Json::Value &req_info,
    const Json::Value &in,
    Json::Value &response,
    struct mg_connection *conn,
    std::function<void(NvDsStreamInfo *stream_ctx, void *ctx)> stream_cb)
{
    NvDsErrorCode ret = NvDsErrorCode::NoError;
    const std::string request_api = req_info.get("url", EMPTY_STRING).asString();
    const std::string request_method = req_info.get("method", UNKNOWN_STRING).asString();
    const std::string query_string = req_info.get("query", EMPTY_STRING).asString();

    if (request_api.empty() || request_method == UNKNOWN_STRING) {
        std::cout << "Malformed HTTP request" << std::endl;
        return NvDsErrorCode::InvalidParameterError;
    }

    if (iequals(request_method, "get")) {
    }

    if (iequals(request_method, "post")) {
        NvDsStreamInfo stream_info;
        NvDsResponseInfo res_info;

        void *custom_ctx;

        if (nvds_rest_stream_parse(in, &stream_info) && (stream_cb)) {
            stream_cb(&stream_info, &custom_ctx);
            if (stream_info.status == STREAM_REMOVE_SUCCESS)
                res_info.status = "STREAM_REMOVE_SUCCESS";
            else
                res_info.status = "STREAM_REMOVE_FAIL";
        } else {
            res_info.status = "STREAM_REMOVE_FAIL";
        }

        res_info.reason = "NA";

        response["status"] = res_info.status;
        response["reason"] = res_info.reason;
    }
    return ret;
}

NvDsRestServer *nvds_rest_server_start(NvDsServerConfig *server_config,
                                       NvDsServerCallbacks *server_cb)
{
    auto roi_cb = server_cb->roi_cb;
    auto dec_cb = server_cb->dec_cb;
    auto stream_cb = server_cb->stream_cb;
    auto infer_cb = server_cb->infer_cb;

    const char *options[] = {"listening_ports", server_config->port.c_str(), 0};

    std::vector<std::string> cpp_options;
    for (long unsigned int i = 0; i < (sizeof(options) / sizeof(options[0]) - 1); i++) {
        cpp_options.push_back(options[i]);
    }

    NvDsRestServer *httpServerHandler = new NvDsRestServer(cpp_options);

    std::map<std::string, NvDsRestServer::httpFunction> m_func;

    m_func["/version"] = [server_cb](const Json::Value &req_info, const Json::Value &in,
                                     Json::Value &out,
                                     struct mg_connection *conn) { return VersionInfo(out, conn); };

    m_func["/roi/update"] = [roi_cb](const Json::Value &req_info, const Json::Value &in,
                                     Json::Value &out, struct mg_connection *conn) {
        return handleUpdateROI(req_info, in, out, conn, roi_cb);
    };

    m_func["/dec/drop-frame-interval"] = [dec_cb](const Json::Value &req_info,
                                                  const Json::Value &in, Json::Value &out,
                                                  struct mg_connection *conn) {
        return handleDecDropFrameInterval(req_info, in, out, conn, dec_cb);
    };

    m_func["/stream/add"] = [stream_cb](const Json::Value &req_info, const Json::Value &in,
                                        Json::Value &out, struct mg_connection *conn) {
        return handleAddStream(req_info, in, out, conn, stream_cb);
    };

    m_func["/stream/remove"] = [stream_cb](const Json::Value &req_info, const Json::Value &in,
                                           Json::Value &out, struct mg_connection *conn) {
        return handleRemoveStream(req_info, in, out, conn, stream_cb);
    };

    m_func["/infer/set-interval"] = [infer_cb](const Json::Value &req_info, const Json::Value &in,
                                               Json::Value &out, struct mg_connection *conn) {
        return handleInferInterval(req_info, in, out, conn, infer_cb);
    };

    for (auto it : m_func) {
        httpServerHandler->addHandler(it.first, new RequestHandler(it.first, it.second));
    }

    std::cout << "Server running at port: " << server_config->port << "\n";

    return httpServerHandler;
}
