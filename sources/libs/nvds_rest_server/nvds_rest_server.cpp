/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: MIT
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
#define stringify(name) #name
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

NvDsErrorCode handleOsdReq(const Json::Value &req_info,
                           const Json::Value &in,
                           Json::Value &response,
                           struct mg_connection *conn,
                           std::function<void(NvDsOsdInfo *osd_ctx, void *ctx)> osd_cb);

NvDsErrorCode handleEncReq(const Json::Value &req_info,
                           const Json::Value &in,
                           Json::Value &response,
                           struct mg_connection *conn,
                           std::function<void(NvDsEncInfo *enc_ctx, void *ctx)> enc_cb);

NvDsErrorCode handleConvReq(const Json::Value &req_info,
                            const Json::Value &in,
                            Json::Value &response,
                            struct mg_connection *conn,
                            std::function<void(NvDsConvInfo *conv_ctx, void *ctx)> conv_cb);

NvDsErrorCode handleMuxReq(const Json::Value &req_info,
                           const Json::Value &in,
                           Json::Value &response,
                           struct mg_connection *conn,
                           std::function<void(NvDsMuxInfo *mux_ctx, void *ctx)> mux_cb);

NvDsErrorCode handleAppReq(
    const Json::Value &req_info,
    const Json::Value &in,
    Json::Value &response,
    struct mg_connection *conn,
    std::function<void(NvDsAppInstanceInfo *appinstance_ctx, void *ctx)> appinstance_cb);

NvDsErrorCode handleDecReq(const Json::Value &req_info,
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

NvDsErrorCode handleInferReq(const Json::Value &req_info,
                             const Json::Value &in,
                             Json::Value &response,
                             struct mg_connection *conn,
                             std::function<void(NvDsInferInfo *infer_ctx, void *ctx)> infer_cb);

NvDsErrorCode handleInferServerReq(
    const Json::Value &req_info,
    const Json::Value &in,
    Json::Value &response,
    struct mg_connection *conn,
    std::function<void(NvDsInferServerInfo *inferserver_ctx, void *ctx)> inferserver_cb);

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
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](char str1, char str2) {
        return std::tolower(str1) == std::tolower(str2);
    });
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
    response["version"] = "DeepStream-SDK 7.0";
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

NvDsErrorCode handleInferReq(const Json::Value &req_info,
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

        if (request_api.find("set-interval") != std::string::npos) {
            infer_info.infer_flag = INFER_INTERVAL;
        }
        if (nvds_rest_infer_parse(in, &infer_info) && (infer_cb)) {
            infer_cb(&infer_info, &custom_ctx);

            switch (infer_info.infer_flag) {
            case INFER_INTERVAL:
                res_info.status = (infer_info.status == INFER_INTERVAL_UPDATE_SUCCESS)
                                      ? "INFER_INTERVAL_UPDATE_SUCCESS"
                                      : "INFER_INTERVAL_UPDATE_FAIL";
                break;
            default:
                break;
            }
        }

        res_info.reason = infer_info.infer_log;

        response["status"] = res_info.status;
        response["reason"] = res_info.reason;
    }

    return ret;
}

NvDsErrorCode handleInferServerReq(
    const Json::Value &req_info,
    const Json::Value &in,
    Json::Value &response,
    struct mg_connection *conn,
    std::function<void(NvDsInferServerInfo *inferserver_ctx, void *ctx)> inferserver_cb)
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
        NvDsInferServerInfo inferserver_info;
        NvDsResponseInfo res_info;

        void *custom_ctx;
        if (request_api.find("set-interval") != std::string::npos) {
            inferserver_info.inferserver_flag = INFERSERVER_INTERVAL;
        }

        if (nvds_rest_inferserver_parse(in, &inferserver_info) && (inferserver_cb)) {
            inferserver_cb(&inferserver_info, &custom_ctx);
            switch (inferserver_info.inferserver_flag) {
            case INFERSERVER_INTERVAL:
                res_info.status = (inferserver_info.status == INFERSERVER_INTERVAL_UPDATE_SUCCESS)
                                      ? "INFERSERVER_INTERVAL_UPDATE_SUCCESS"
                                      : "INFERSERVER_INTERVAL_UPDATE_FAIL";
                break;
            default:
                break;
            }
        }

        res_info.reason = inferserver_info.inferserver_log;

        response["status"] = res_info.status;
        response["reason"] = res_info.reason;
    }

    return ret;
}

NvDsErrorCode handleDecReq(const Json::Value &req_info,
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
        if (request_api.find("drop-frame-interval") != std::string::npos) {
            dec_info.dec_flag = DROP_FRAME_INTERVAL;
        }
        if (request_api.find("skip-frames") != std::string::npos) {
            dec_info.dec_flag = SKIP_FRAMES;
        }
        if (request_api.find("low-latency-mode") != std::string::npos) {
            dec_info.dec_flag = LOW_LATENCY_MODE;
        }

        if (nvds_rest_dec_parse(in, &dec_info) && (dec_cb)) {
            dec_cb(&dec_info, &custom_ctx);
            switch (dec_info.dec_flag) {
            case DROP_FRAME_INTERVAL:
                res_info.status = (dec_info.status == DROP_FRAME_INTERVAL_UPDATE_SUCCESS)
                                      ? "DROP_FRAME_INTERVAL_UPDATE_SUCCESS"
                                      : "DROP_FRAME_INTERVAL_UPDATE_FAIL";
                break;
            case SKIP_FRAMES:
                res_info.status = (dec_info.status == SKIP_FRAMES_UPDATE_SUCCESS)
                                      ? "SKIP_FRAMES_UPDATE_SUCCESS"
                                      : "SKIP_FRAMES_UPDATE_FAIL";
                break;
            case LOW_LATENCY_MODE:
                res_info.status = (dec_info.status == LOW_LATENCY_MODE_UPDATE_SUCCESS)
                                      ? "LOW_LATENCY_MODE_UPDATE_SUCCESS"
                                      : "LOW_LATENCY_MODE_UPDATE_FAIL";
                break;
            default:
                break;
            }
        }

        res_info.reason = dec_info.dec_log;

        response["status"] = res_info.status;
        response["reason"] = res_info.reason;
    }

    return ret;
}

NvDsErrorCode handleEncReq(const Json::Value &req_info,
                           const Json::Value &in,
                           Json::Value &response,
                           struct mg_connection *conn,
                           std::function<void(NvDsEncInfo *enc_ctx, void *ctx)> enc_cb)
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
        NvDsEncInfo enc_info;
        NvDsResponseInfo res_info;

        void *custom_ctx;
        if (request_api.find("bitrate") != std::string::npos) {
            enc_info.enc_flag = BITRATE;
        }
        if (request_api.find("force-idr") != std::string::npos) {
            enc_info.enc_flag = FORCE_IDR;
        }
        if (request_api.find("force-intra") != std::string::npos) {
            enc_info.enc_flag = FORCE_INTRA;
        }
        if (request_api.find("iframe-interval") != std::string::npos) {
            enc_info.enc_flag = IFRAME_INTERVAL;
        }
        if (nvds_rest_enc_parse(in, &enc_info) && (enc_cb)) {
            enc_cb(&enc_info, &custom_ctx);
            switch (enc_info.enc_flag) {
            case BITRATE:
                res_info.status = (enc_info.status == BITRATE_UPDATE_SUCCESS)
                                      ? "BITRATE_UPDATE_SUCCESS"
                                      : "BITRATE_UPDATE_FAIL";
                break;
            case FORCE_IDR:
                res_info.status = (enc_info.status == FORCE_IDR_UPDATE_SUCCESS)
                                      ? "FORCE_IDR_UPDATE_SUCCESS"
                                      : "FORCE_IDR_UPDATE_UPDATE_FAIL";
                break;
            case FORCE_INTRA:
                res_info.status = (enc_info.status == FORCE_INTRA_UPDATE_SUCCESS)
                                      ? "FORCE_INTRA_UPDATE_SUCCESS"
                                      : "FORCE_INTRA_UPDATE_FAIL";
                break;
            case IFRAME_INTERVAL:
                res_info.status = (enc_info.status == IFRAME_INTERVAL_UPDATE_SUCCESS)
                                      ? "IFRAME_INTERVAL_UPDATE_SUCCESS"
                                      : "IFRAME_INTERVAL_UPDATE_FAIL";
                break;
            default:
                break;
            }
        }

        res_info.reason = enc_info.enc_log;

        response["status"] = res_info.status;
        response["reason"] = res_info.reason;
    }

    return ret;
}

NvDsErrorCode handleConvReq(const Json::Value &req_info,
                            const Json::Value &in,
                            Json::Value &response,
                            struct mg_connection *conn,
                            std::function<void(NvDsConvInfo *conv_ctx, void *ctx)> conv_cb)
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
        NvDsConvInfo conv_info;
        NvDsResponseInfo res_info;

        void *custom_ctx;
        if (request_api.find("srccrop") != std::string::npos) {
            conv_info.conv_flag = SRC_CROP;
        }
        if (request_api.find("destcrop") != std::string::npos) {
            conv_info.conv_flag = DEST_CROP;
        }
        if (request_api.find("flip-method") != std::string::npos) {
            conv_info.conv_flag = FLIP_METHOD;
        }
        if (request_api.find("interpolation-method") != std::string::npos) {
            conv_info.conv_flag = INTERPOLATION_METHOD;
        }
        if (nvds_rest_conv_parse(in, &conv_info) && (conv_cb)) {
            conv_cb(&conv_info, &custom_ctx);
            switch (conv_info.conv_flag) {
            case SRC_CROP:
                res_info.status = (conv_info.status == SRC_CROP_UPDATE_SUCCESS)
                                      ? "SRC_CROP_UPDATE_SUCCESS"
                                      : "SRC_CROP_UPDATE_FAIL";
                break;
            case DEST_CROP:
                res_info.status = (conv_info.status == DEST_CROP_UPDATE_SUCCESS)
                                      ? "DEST_CROP_UPDATE_SUCCESS"
                                      : "DEST_CROP_UPDATE_UPDATE_FAIL";
                break;
            case FLIP_METHOD:
                res_info.status = (conv_info.status == FLIP_METHOD_UPDATE_SUCCESS)
                                      ? "FLIP_METHOD_UPDATE_SUCCESS"
                                      : "FLIP_METHOD_UPDATE_FAIL";
                break;
            case INTERPOLATION_METHOD:
                res_info.status = (conv_info.status == INTERPOLATION_METHOD_UPDATE_SUCCESS)
                                      ? "INTERPOLATION_METHOD_UPDATE_SUCCESS"
                                      : "INTERPOLATION_METHOD_UPDATE_FAIL";
                break;
            default:
                break;
            }
        }

        res_info.reason = conv_info.conv_log;

        response["status"] = res_info.status;
        response["reason"] = res_info.reason;
    }

    return ret;
}

NvDsErrorCode handleMuxReq(const Json::Value &req_info,
                           const Json::Value &in,
                           Json::Value &response,
                           struct mg_connection *conn,
                           std::function<void(NvDsMuxInfo *mux_ctx, void *ctx)> mux_cb)
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
        NvDsMuxInfo mux_info;
        NvDsResponseInfo res_info;

        if (request_api.find("batched-push-timeout") != std::string::npos) {
            mux_info.mux_flag = BATCHED_PUSH_TIMEOUT;
        }
        if (request_api.find("max-latency") != std::string::npos) {
            mux_info.mux_flag = MAX_LATENCY;
        }

        void *custom_ctx;
        if (nvds_rest_mux_parse(in, &mux_info) && (mux_cb)) {
            mux_cb(&mux_info, &custom_ctx);
            switch (mux_info.mux_flag) {
            case BATCHED_PUSH_TIMEOUT:
                res_info.status = (mux_info.status == BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS)
                                      ? "BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS"
                                      : "BATCHED_PUSH_TIMEOUT_UPDATE_FAIL";
                break;
            case MAX_LATENCY:
                res_info.status = (mux_info.status == MAX_LATENCY_UPDATE_SUCCESS)
                                      ? "MAX_LATENCY_UPDATE_SUCCESS"
                                      : "MAX_LATENCY_UPDATE_FAIL";
                break;
            default:
                break;
            }
        }

        res_info.reason = mux_info.mux_log;

        response["status"] = res_info.status;
        response["reason"] = res_info.reason;
    }

    return ret;
}

NvDsErrorCode handleOsdReq(const Json::Value &req_info,
                           const Json::Value &in,
                           Json::Value &response,
                           struct mg_connection *conn,
                           std::function<void(NvDsOsdInfo *osd_ctx, void *ctx)> osd_cb)
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
        NvDsOsdInfo osd_info;
        NvDsResponseInfo res_info;

        if (request_api.find("process-mode") != std::string::npos) {
            osd_info.osd_flag = PROCESS_MODE;
        }

        void *custom_ctx;
        if (nvds_rest_osd_parse(in, &osd_info) && (osd_cb)) {
            osd_cb(&osd_info, &custom_ctx);
            switch (osd_info.osd_flag) {
            case PROCESS_MODE:
                res_info.status = (osd_info.status == PROCESS_MODE_UPDATE_SUCCESS)
                                      ? "PROCESS_MODE_UPDATE_SUCCESS"
                                      : "PROCESS_MODE_UPDATE_FAIL";
                break;
            default:
                break;
            }
        }

        res_info.reason = osd_info.osd_log;

        response["status"] = res_info.status;
        response["reason"] = res_info.reason;
    }

    return ret;
}

NvDsErrorCode handleAppReq(
    const Json::Value &req_info,
    const Json::Value &in,
    Json::Value &response,
    struct mg_connection *conn,
    std::function<void(NvDsAppInstanceInfo *appinstance_ctx, void *ctx)> appinstance_cb)
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
        NvDsAppInstanceInfo appinstance_info;
        NvDsResponseInfo res_info;

        if (request_api.find("quit") != std::string::npos) {
            appinstance_info.appinstance_flag = QUIT_APP;
        }

        void *custom_ctx;
        if (nvds_rest_app_instance_parse(in, &appinstance_info) && (appinstance_cb)) {
            appinstance_cb(&appinstance_info, &custom_ctx);

            switch (appinstance_info.appinstance_flag) {
            case QUIT_APP:
                res_info.status =
                    (appinstance_info.status == QUIT_SUCCESS) ? "QUIT_SUCCESS" : "QUIT_FAIL";
                break;
            default:
                break;
            }
        }

        res_info.reason = appinstance_info.app_log;
        if (res_info.reason == "")
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

        if (request_api.find("update") != std::string::npos) {
            roi_info.roi_flag = ROI_UPDATE;
        }
        if (nvds_rest_roi_parse(in, &roi_info) && (roi_cb)) {
            roi_cb(&roi_info, &custom_ctx);
            switch (roi_info.roi_flag) {
            case ROI_UPDATE:
                res_info.status = (roi_info.status == ROI_UPDATE_SUCCESS) ? "ROI_UPDATE_SUCCESS"
                                                                          : "ROI_UPDATE_FAIL";
                break;
            default:
                break;
            }
        }

        res_info.reason = roi_info.roi_log;

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
        res_info.reason = stream_info.stream_log;

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

        res_info.reason = stream_info.stream_log;

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
    auto enc_cb = server_cb->enc_cb;
    auto stream_cb = server_cb->stream_cb;
    auto infer_cb = server_cb->infer_cb;
    auto conv_cb = server_cb->conv_cb;
    auto mux_cb = server_cb->mux_cb;
    auto inferserver_cb = server_cb->inferserver_cb;
    auto osd_cb = server_cb->osd_cb;
    auto appinstance_cb = server_cb->appinstance_cb;

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

    /* Stream Management Specific */
    m_func["/stream/add"] = [stream_cb](const Json::Value &req_info, const Json::Value &in,
                                        Json::Value &out, struct mg_connection *conn) {
        return handleAddStream(req_info, in, out, conn, stream_cb);
    };

    m_func["/stream/remove"] = [stream_cb](const Json::Value &req_info, const Json::Value &in,
                                           Json::Value &out, struct mg_connection *conn) {
        return handleRemoveStream(req_info, in, out, conn, stream_cb);
    };

    /* Pre-Process Specific */
    m_func["/roi/update"] = [roi_cb](const Json::Value &req_info, const Json::Value &in,
                                     Json::Value &out, struct mg_connection *conn) {
        return handleUpdateROI(req_info, in, out, conn, roi_cb);
    };

    /* Decoder Specific */
    m_func["/dec/drop-frame-interval"] = [dec_cb](const Json::Value &req_info,
                                                  const Json::Value &in, Json::Value &out,
                                                  struct mg_connection *conn) {
        return handleDecReq(req_info, in, out, conn, dec_cb);
    };

    m_func["/dec/skip-frames"] = [dec_cb](const Json::Value &req_info, const Json::Value &in,
                                          Json::Value &out, struct mg_connection *conn) {
        return handleDecReq(req_info, in, out, conn, dec_cb);
    };

    m_func["/dec/low-latency-mode"] = [dec_cb](const Json::Value &req_info, const Json::Value &in,
                                               Json::Value &out, struct mg_connection *conn) {
        return handleDecReq(req_info, in, out, conn, dec_cb);
    };

    /* Encoder Specific */
    m_func["/enc/bitrate"] = [enc_cb](const Json::Value &req_info, const Json::Value &in,
                                      Json::Value &out, struct mg_connection *conn) {
        return handleEncReq(req_info, in, out, conn, enc_cb);
    };

    m_func["/enc/force-idr"] = [enc_cb](const Json::Value &req_info, const Json::Value &in,
                                        Json::Value &out, struct mg_connection *conn) {
        return handleEncReq(req_info, in, out, conn, enc_cb);
    };

    m_func["/enc/force-intra"] = [enc_cb](const Json::Value &req_info, const Json::Value &in,
                                          Json::Value &out, struct mg_connection *conn) {
        return handleEncReq(req_info, in, out, conn, enc_cb);
    };

    m_func["/enc/iframe-interval"] = [enc_cb](const Json::Value &req_info, const Json::Value &in,
                                              Json::Value &out, struct mg_connection *conn) {
        return handleEncReq(req_info, in, out, conn, enc_cb);
    };

    /* Inference Specific */
    m_func["/infer/set-interval"] = [infer_cb](const Json::Value &req_info, const Json::Value &in,
                                               Json::Value &out, struct mg_connection *conn) {
        return handleInferReq(req_info, in, out, conn, infer_cb);
    };

    m_func["/inferserver/set-interval"] = [inferserver_cb](const Json::Value &req_info,
                                                           const Json::Value &in, Json::Value &out,
                                                           struct mg_connection *conn) {
        return handleInferServerReq(req_info, in, out, conn, inferserver_cb);
    };

    /* video convert Specific */
    m_func["/conv/destcrop"] = [conv_cb](const Json::Value &req_info, const Json::Value &in,
                                         Json::Value &out, struct mg_connection *conn) {
        return handleConvReq(req_info, in, out, conn, conv_cb);
    };

    m_func["/conv/srccrop"] = [conv_cb](const Json::Value &req_info, const Json::Value &in,
                                        Json::Value &out, struct mg_connection *conn) {
        return handleConvReq(req_info, in, out, conn, conv_cb);
    };

    m_func["/conv/interpolation-method"] = [conv_cb](const Json::Value &req_info,
                                                     const Json::Value &in, Json::Value &out,
                                                     struct mg_connection *conn) {
        return handleConvReq(req_info, in, out, conn, conv_cb);
    };

    m_func["/conv/flip-method"] = [conv_cb](const Json::Value &req_info, const Json::Value &in,
                                            Json::Value &out, struct mg_connection *conn) {
        return handleConvReq(req_info, in, out, conn, conv_cb);
    };

    m_func["/mux/batched-push-timeout"] = [mux_cb](const Json::Value &req_info,
                                                   const Json::Value &in, Json::Value &out,
                                                   struct mg_connection *conn) {
        return handleMuxReq(req_info, in, out, conn, mux_cb);
    };

    m_func["/mux/max-latency"] = [mux_cb](const Json::Value &req_info, const Json::Value &in,
                                          Json::Value &out, struct mg_connection *conn) {
        return handleMuxReq(req_info, in, out, conn, mux_cb);
    };

    m_func["/osd/process-mode"] = [osd_cb](const Json::Value &req_info, const Json::Value &in,
                                           Json::Value &out, struct mg_connection *conn) {
        return handleOsdReq(req_info, in, out, conn, osd_cb);
    };

    m_func["/app/quit"] = [appinstance_cb](const Json::Value &req_info, const Json::Value &in,
                                           Json::Value &out, struct mg_connection *conn) {
        return handleAppReq(req_info, in, out, conn, appinstance_cb);
    };

    for (auto it : m_func) {
        httpServerHandler->addHandler(it.first, new RequestHandler(it.first, it.second));
    }

    std::cout << "Server running at port: " << server_config->port << "\n";

    return httpServerHandler;
}
